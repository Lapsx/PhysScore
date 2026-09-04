import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool

class GaussianSmearing(nn.Module):
    """
    Expande a distância escalar Euclidiana em um conjunto contínuo de Funções de Base Radial (RBF).
    Isso é crucial para invariância SE(3), mapeando distâncias físicas para um manifold suave.
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=32):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ContinuousFilterConv(MessagePassing):
    """
    Convolução de Filtro Contínuo (inspirado no SchNet).
    A distância geométrica RBF modula as matrizes de pesos, evitando a concatenação ingênua.
    """
    def __init__(self, node_dim=64, num_gaussians=32):
        super(ContinuousFilterConv, self).__init__(aggr='add')
        
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # O filtro contínuo W atua como um portão (gate) modulado pela distância espacial
        W = self.filter_network(edge_attr)
        return x_j * W

    def update(self, aggr_out, x):
        # Conexão residual estrita (ResNet style) para evitar vanishing gradients
        return x + self.update_mlp(aggr_out)

class NodeEncoder(nn.Module):
    """Codifica cada átomo a partir de Z, das features químicas e do seu papel.

    O canal de papel (ligante ou proteína) não existia: `to_homogeneous()` funde os
    dois tipos de nó e o modelo via uma nuvem indiferenciada de átomos, sem saber
    qual parte é o fármaco. Para prever afinidade de ligação isso é informação
    central, e ela já estava disponível em `data.node_type` — apenas não era usada.
    """
    def __init__(self, max_z=100, z_emb_dim=32, phys_feat_dim=12, out_dim=64,
                 dropout=0.2, papel_emb_dim=8):
        super().__init__()
        self.z_embedding = nn.Embedding(max_z, z_emb_dim)
        self.papel_embedding = nn.Embedding(2, papel_emb_dim)   # 0 = ligante, 1 = proteína
        self.fusion_mlp = nn.Sequential(
            nn.Linear(z_emb_dim + phys_feat_dim + papel_emb_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_raw, node_type=None):
        # x_raw[:, 0] é o Número Atômico Z (índice categórico)
        # x_raw[:, 1:] são as 12 features químicas restantes
        z_idx = x_raw[:, 0].long()
        phys_features = x_raw[:, 1:]

        z_emb = self.z_embedding(z_idx)

        if node_type is None:   # compatibilidade com chamadas antigas
            node_type = torch.zeros(x_raw.shape[0], dtype=torch.long, device=x_raw.device)
        papel_emb = self.papel_embedding(node_type.long().clamp(0, 1))

        h_0 = torch.cat([z_emb, phys_features, papel_emb], dim=-1)
        return self.fusion_mlp(h_0)

class ReadoutPorPapel(nn.Module):
    """Readout que separa ligante de proteína e preserva a extensividade.

    O readout anterior era um `global_mean_pool` sobre a união dos átomos, e
    isso custava caro por duas razões medidas em 04/09/2026:

    1. **Diluição.** O ligante tem 33 átomos e o bolso 130 (razão 1:3.9). Numa
       média sobre a união, o fármaco — a parte que de fato varia entre os
       complexos de um mesmo alvo — entra com peso 1/4.

    2. **A média destrói o preditor mais forte disponível.** Afinidade cresce com
       o número de contatos favoráveis: `log(n_contatos)` sozinho correlaciona
       +0.349 com pKd no conjunto de treino. Uma média normaliza exatamente por
       tamanho, de modo que o modelo precisava reconstruir esse sinal por vias
       indiretas. A soma o entrega de graça.

    O diagnóstico que motivou a troca: uma regressão linear sobre 8 features de
    contagem atinge R = 0.592 no CASF-2016 core, contra R = 0.680 do GNN inteiro.
    Toda a maquinaria geométrica valia 0.088 de R. O gargalo era o readout, não
    falta de regularização — o R de *treino* em modo eval era 0.720, longe da
    memorização que o gap da loss sugeria.

    A soma passa por LayerNorm porque escala com o tamanho do grafo (90 a 256
    átomos) e alimentaria o MLP com ativações de magnitude muito variável.
    """
    def __init__(self, node_dim=64):
        super().__init__()
        self.norm_soma_lig = nn.LayerNorm(node_dim)
        self.norm_soma_prot = nn.LayerNorm(node_dim)
        # 4 vetores de node_dim (média e soma, para cada papel) + 3 descritores
        self.out_dim = node_dim * 4 + 3

    def forward(self, x, batch, node_type, num_graphs, n_contatos):
        m_lig = (node_type == 0).float().unsqueeze(-1)
        m_prot = (node_type == 1).float().unsqueeze(-1)

        soma_lig = global_add_pool(x * m_lig, batch, size=num_graphs)
        soma_prot = global_add_pool(x * m_prot, batch, size=num_graphs)
        n_lig = global_add_pool(m_lig, batch, size=num_graphs).clamp(min=1.0)
        n_prot = global_add_pool(m_prot, batch, size=num_graphs).clamp(min=1.0)

        descritores = torch.cat(
            [torch.log1p(n_lig), torch.log1p(n_prot), torch.log1p(n_contatos)], dim=-1)

        return torch.cat([
            soma_lig / n_lig,
            soma_prot / n_prot,
            self.norm_soma_lig(soma_lig),
            self.norm_soma_prot(soma_prot),
            descritores,
        ], dim=-1)


class PharmGeometricGNN(nn.Module):
    """GNN geométrica de afinidade.

    `usar_assay` liga um embedding de tipo de ensaio experimental. Motivo: 51,3%
    do conjunto de treino (general set) é IC50, enquanto validação e teste
    (refined e CASF core) têm 0% — o refined exclui IC50 por critério de
    qualidade. IC50 depende das condições do ensaio (concentração de substrato,
    pH, tempo) e não é constante termodinâmica, então metade do sinal de treino
    carrega um viés sistemático que não existe onde o modelo é avaliado. Dar ao
    modelo um token do tipo de ensaio permite que ele aprenda esse viés como um
    deslocamento explícito em vez de absorvê-lo nos pesos compartilhados.

    O embedding fica desligado por padrão para não alterar o `state_dict` de
    quem carrega os checkpoints existentes.
    """

    # Ordem dos tipos de ensaio. Kd é o índice 0 e vira o padrão na inferência:
    # é a constante termodinâmica que se quer prever, e é o regime do CASF.
    TIPOS_ENSAIO = ('Kd', 'Ki', 'IC50', 'desconhecido')

    def __init__(self, node_dim=64, num_gaussians=32, num_layers=4, dropout=0.2,
                 usar_assay=False, assay_emb_dim=8):
        super(PharmGeometricGNN, self).__init__()
        
        self.node_encoder = NodeEncoder(max_z=100, z_emb_dim=32, phys_feat_dim=12, out_dim=node_dim, dropout=dropout)
        self.rbf_expansion = GaussianSmearing(start=0.0, stop=10.0, num_gaussians=num_gaussians)
        
        self.layers = nn.ModuleList([
            ContinuousFilterConv(node_dim, num_gaussians) for _ in range(num_layers)
        ])
        
        self.readout = ReadoutPorPapel(node_dim)
        d_read = self.readout.out_dim

        self.usar_assay = usar_assay
        if usar_assay:
            self.assay_embedding = nn.Embedding(len(self.TIPOS_ENSAIO), assay_emb_dim)
            d_read += assay_emb_dim

        # Predictor Primário: Afinidade de Ligação (pKd)
        self.predictor = nn.Sequential(
            nn.Linear(d_read, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Predictor Auxiliar: Regularização Física (Potencial Estérico)
        self.steric_predictor = nn.Sequential(
            nn.Linear(d_read, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, data_homo, assay=None):
        x, edge_index, pos, batch = data_homo.x, data_homo.edge_index, data_homo.pos, data_homo.batch
        
        # Se 'pos' exige gradiente (Fase de Relaxamento PINN), computa dinamicamente
        # Caso contrário (Fase de Explicação), usa edge_attr explícito para Saliency Map
        if pos.requires_grad:
            row, col = edge_index
            edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1).view(-1, 1)
        else:
            edge_attr = data_homo.edge_attr
        
        # 1. Encoding Inicial Hibrido (Z + Features Químicas + papel ligante/proteína)
        x = self.node_encoder(x, getattr(data_homo, 'node_type', None))
        
        # 2. Expansão Contínua de Distâncias (Invariância Translacional e Rotacional)
        rbf_attr = self.rbf_expansion(edge_attr)
        
        # 3. Propagação Geométrica Modulada
        for layer in self.layers:
            x = layer(x, edge_index, rbf_attr)
            
        # 4. Readout por papel (ligante x proteína), com média e soma
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1

        node_type = getattr(data_homo, 'node_type', None)
        if node_type is None:
            node_type = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Contatos não-ligados por grafo: em data_processor.py os tipos 0 e 1 são
        # covalentes e 2 e 3 são de interação. Sem edge_type, separa pela geometria
        # (o raio covalente usado na construção do grafo é 2.0 A), a mesma regra que
        # calculate_lj_potential aplica em train.py.
        edge_type = getattr(data_homo, 'edge_type', None)
        if edge_type is not None:
            mascara_contato = (edge_type >= 2)
        else:
            mascara_contato = (edge_attr.flatten() > 2.0)
        n_contatos = torch.zeros(num_graphs, 1, device=x.device).scatter_add_(
            0,
            batch[edge_index[0][mascara_contato]].view(-1, 1),
            torch.ones(int(mascara_contato.sum()), 1, device=x.device))

        x_graph = self.readout(x, batch, node_type, num_graphs, n_contatos)

        if self.usar_assay:
            # Sem tipo informado (inferência em complexo novo), assume Kd: é o
            # alvo termodinâmico desejado, não o viés de ensaio de um IC50.
            if assay is None:
                assay = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            x_graph = torch.cat([x_graph, self.assay_embedding(assay.view(-1).long())], dim=-1)
        
        # 5. Saídas Multi-Tarefa
        pkd_pred = self.predictor(x_graph)
        steric_pred = self.steric_predictor(x_graph)
        
        return pkd_pred, steric_pred
