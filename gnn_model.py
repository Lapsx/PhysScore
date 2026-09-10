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

class CamadaDirecional(nn.Module):
    """Bloco de mensagem + atualização com features vetoriais (estilo PaiNN).

    **Por que existe.** As convoluções de filtro contínuo acima veem apenas
    `||r_i - r_j||`. Distância não é ângulo, e a consequência é medida: uma ligação
    de hidrogênio N-H···O vale muito a 180° e quase nada a 90°, com a MESMA
    distância — o modelo não consegue distinguir os dois casos.

    Isso aparece nas duas métricas do projeto, sempre na mesma metade:

    | tarefa | distância basta | exige ângulo |
    |---|---|---|
    | afinidade | ordenar (rho 0,746) | acertar magnitude (slope 0,424) |
    | pose | plausível vs. absurda (rho global 0,374) | escolher entre plausíveis (0,249) |

    Nos dois casos o que falha é a discriminação fina. Não é coincidência.

    **Como preserva a invariância.** Cada átomo carrega escalares `s` [N, F] e
    vetores `v` [N, 3, F]. Os vetores giram junto com o complexo (equivariância),
    mas só entram em `s` através de quantidades invariantes — a norma `||Vv||` e o
    produto interno `<Uv, Vv>`. Como o readout consome apenas `s`, a saída continua
    exatamente invariante a rotação e translação, como antes.

    **Por que PaiNN e não DimeNet.** Capturar ângulo enumerando tripletos custa
    O(E·k) e mataria o argumento central do projeto, que é fazer isto com ~127 mil
    parâmetros. Aqui a direcionalidade emerge do produto interno entre vetores, sem
    enumerar tripleto nenhum.
    """

    def __init__(self, node_dim=64, num_gaussians=32):
        super().__init__()
        self.F = node_dim
        self.phi = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim * 3),
        )
        self.W = nn.Linear(num_gaussians, node_dim * 3)
        self.U = nn.Linear(node_dim, node_dim, bias=False)
        self.V = nn.Linear(node_dim, node_dim, bias=False)
        self.mlp_upd = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim * 3),
        )

    def forward(self, s, v, edge_index, rbf, versor):
        F = self.F
        j, i = edge_index[0], edge_index[1]

        # ---- mensagem ----
        filtro = self.W(rbf) * self.phi(s[j])            # [E, 3F]
        f_s, f_vv, f_vs = filtro.split(F, dim=-1)

        s = s + torch.zeros_like(s).index_add_(0, i, f_s)
        msg_v = v[j] * f_vv.unsqueeze(1) + versor.unsqueeze(-1) * f_vs.unsqueeze(1)
        v = v + torch.zeros_like(v).index_add_(0, i, msg_v)

        # ---- atualização ----
        Uv, Vv = self.U(v), self.V(v)                    # [N, 3, F]
        # O epsilon evita gradiente infinito quando um canal vetorial zera, o que
        # acontece de fato na primeira camada (v começa em zero). Com 1e-8 a
        # primeira derivada já vale 5.000 e a segunda ~1e11; sob double backward
        # isso basta para desestabilizar. 1e-6 mantém a derivada abaixo de 500.
        norma_Vv = torch.sqrt((Vv ** 2).sum(dim=1) + 1e-6)
        a_vv, a_sv, a_ss = self.mlp_upd(
            torch.cat([s, norma_Vv], dim=-1)).split(F, dim=-1)

        v = v + Uv * a_vv.unsqueeze(1)
        s = s + a_ss + a_sv * (Uv * Vv).sum(dim=1)       # <Uv, Vv> é invariante
        return s, v


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
                 usar_assay=False, assay_emb_dim=8, usar_pose=False,
                 usar_direcional=False):
        super(PharmGeometricGNN, self).__init__()
        
        self.node_encoder = NodeEncoder(max_z=100, z_emb_dim=32, phys_feat_dim=12, out_dim=node_dim, dropout=dropout)
        self.rbf_expansion = GaussianSmearing(start=0.0, stop=10.0, num_gaussians=num_gaussians)
        
        # `usar_direcional` troca as convoluções de filtro contínuo (só distância)
        # por blocos com features vetoriais equivariantes (distância + direção).
        # Desligado por padrão: o state_dict muda por completo, e os checkpoints
        # existentes são de camadas escalares.
        self.usar_direcional = usar_direcional
        self.node_dim = node_dim
        if usar_direcional:
            self.layers = nn.ModuleList([
                CamadaDirecional(node_dim, num_gaussians) for _ in range(num_layers)
            ])
        else:
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

        # Predictor de Qualidade de Pose — separado do de afinidade, de propósito.
        #
        # Medido em 09/09/2026: o modelo é 2º de 35 em scoring power e 30º de 35 em
        # docking power. O déficit não está na tendência global (rho 0,374 contra
        # 0,334 do AutodockVina, à frente), e sim na vizinhança da pose nativa
        # (0,249 contra 0,614). Ver a seção 4 do EXPERIMENTOS.md.
        #
        # Por que uma cabeça própria e não o pKd: pKd é energia LIVRE, e a pose é
        # determinada por energia POTENCIAL. As duas tarefas puxam em direções
        # diferentes — o AutodockVina faz 90,1% de top1 com R = 0,604, contra 47,0%
        # e R = 0,754 aqui. Impor comportamento de energia sobre a cabeça de
        # afinidade arriscaria o 0,754 sem necessidade.
        self.usar_pose = usar_pose
        if usar_pose:
            self.pose_predictor = nn.Sequential(
                nn.Linear(d_read, 64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def forward(self, data_homo, assay=None, retornar_pose=False):
        x, edge_index, pos, batch = data_homo.x, data_homo.edge_index, data_homo.pos, data_homo.batch
        
        # Se 'pos' exige gradiente (estacionariedade, relaxamento), computa
        # dinamicamente. Caso contrário (Saliency Map), usa edge_attr explícito.
        # No modo direcional o vetor da aresta é sempre necessário, então as
        # distâncias saem dele em vez do edge_attr guardado.
        versor = None
        if self.usar_direcional:
            j, i = edge_index[0], edge_index[1]
            vetor = pos[i] - pos[j]
            # O piso é FÍSICO, não numérico. A derivada segunda do versor escala
            # com 1/d², e a estacionariedade da cabeça de pose usa double backward:
            # com o piso em 1e-6 isso chega a 1e12 e o treino vira NaN na 6ª época
            # (medido — a loss de estacionariedade foi a 703.684 antes de estourar).
            #
            # E a distância zero acontece de fato: 72 arestas em 3,7 milhões medem
            # menos de 1e-6 Å, afetando 1,1% dos grafos — átomos sobrepostos que o
            # PDB traz por conformação alternativa. Nenhuma ligação química real
            # mede menos que ~0,9 Å, então 0,5 Å descarta só artefato. No caminho
            # escalar isso passa batido porque o RBF satura em vez de explodir.
            dist = torch.norm(vetor, p=2, dim=1).clamp(min=0.5)
            versor = vetor / dist.unsqueeze(-1)
            edge_attr = dist.view(-1, 1)
        elif pos.requires_grad:
            row, col = edge_index
            edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1).view(-1, 1)
        else:
            edge_attr = data_homo.edge_attr
        
        # 1. Encoding Inicial Hibrido (Z + Features Químicas + papel ligante/proteína)
        x = self.node_encoder(x, getattr(data_homo, 'node_type', None))
        
        # 2. Expansão Contínua de Distâncias (Invariância Translacional e Rotacional)
        rbf_attr = self.rbf_expansion(edge_attr)
        
        # 3. Propagação Geométrica Modulada
        if self.usar_direcional:
            # v começa em zero: sem direção privilegiada antes da primeira mensagem
            v = torch.zeros(x.shape[0], 3, self.node_dim,
                            device=x.device, dtype=x.dtype)
            for layer in self.layers:
                x, v = layer(x, v, edge_index, rbf_attr, versor)
        else:
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

        # A assinatura de 2 saídas é preservada por padrão: `explain.py`,
        # `predict_custom.py` e `consenso_farmacoforo.py` desempacotam dois valores.
        if retornar_pose:
            pose_pred = self.pose_predictor(x_graph) if self.usar_pose else None
            return pkd_pred, steric_pred, pose_pred

        return pkd_pred, steric_pred
