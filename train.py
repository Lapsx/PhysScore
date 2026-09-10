import torch
import numpy as np
import gc
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import re
import random

from dataset import PDBbindDataset
from gnn_model import PharmGeometricGNN

def load_refined_ids(filepath="refined_ids.txt"):
    """Lê o arquivo gerado pelo bash script com os IDs oficiais de Validação/Teste."""
    if not os.path.exists(filepath):
        raise FileNotFoundError("O arquivo refined_ids.txt não foi encontrado! Rode o flatten_raw.sh primeiro.")

    with open(filepath, 'r') as f:
        return set(line.strip().lower() for line in f if line.strip())


def load_core_ids(filepath="core_set.dat"):
    """Lê o CoreSet.dat do pacote CASF-2016 (285 complexos em 57 clusters).

    Este é o conjunto de TESTE. A distinção em relação à validação é essencial e
    não é burocracia: o checkpoint é escolhido pela loss de validação, de modo que
    qualquer número medido nela é otimista por construção — o conjunto participou
    da seleção do modelo. O core set nunca é consultado durante o treino nem na
    escolha do checkpoint, e por isso a métrica sobre ele é a única reportável.

    É também o conjunto onde a literatura da área publica, o que torna o resultado
    diretamente comparável a outros métodos.
    """
    if not os.path.exists(filepath):
        print(f"[!] {filepath} não encontrado: seguindo sem conjunto de teste.")
        print("    Extraia CASF-2016/power_screening/CoreSet.dat do pacote CASF-2016.")
        return set()

    ids = set()
    with open(filepath, 'r') as f:
        for linha in f:
            if linha.startswith('#'):
                continue
            partes = linha.split()
            if partes:
                ids.add(partes[0].strip().lower())
    return ids


def carrega_tipo_ensaio(filepath="index/INDEX_general_PL.2020R1.lst"):
    """Mapeia pdb_id -> tipo de medida experimental (Kd, Ki ou IC50).

    O formato do INDEX do PDBbind 2020 é
    `pdb  resolucao  ano  Kd=49uM  // referencia (ligante)`, portanto o tipo está
    no quarto campo, colado ao valor. Contagem global: 7.043 Kd, 4.961 Ki e
    7.033 IC50 nos 19.037 complexos.
    """
    tipos = {}
    if not os.path.exists(filepath):
        print(f"[!] {filepath} nao encontrado: tipo de ensaio indisponivel.")
        return tipos
    with open(filepath) as f:
        for linha in f:
            if linha.startswith('#'):
                continue
            partes = linha.split()
            if len(partes) < 4:
                continue
            m = re.match(r'(IC50|Kd|Ki)', partes[3])
            if m:
                tipos[partes[0].lower()] = m.group(1)
    return tipos


# Índices dos tipos de aresta em HeteroData.to_homogeneous(), na ordem em que
# data_processor.py os cria: 0 e 1 são covalentes, 2 e 3 são de interação.
TIPOS_NAO_LIGADOS = 2


# ----------------------------------------------------------------------------
# Física de pose (seção 4 do EXPERIMENTOS.md)
# ----------------------------------------------------------------------------

def _centroides_ligante(pos, mascara_lig, batch, num_graphs):
    """Centroide do ligante de cada grafo, e o índice de grafo de cada átomo dele."""
    b = batch[mascara_lig]
    p = pos[mascara_lig]
    contagem = torch.zeros(num_graphs, 1, device=pos.device).index_add_(
        0, b, torch.ones(b.shape[0], 1, device=pos.device))
    centro = torch.zeros(num_graphs, 3, device=pos.device).index_add_(0, b, p)
    return centro / contagem.clamp(min=1.0), b, p, contagem


def perturbar_ligante(pos, node_type, batch, num_graphs,
                      desloc_min=0.3, desloc_max=3.0, ang_max=0.35):
    """Move o ligante como corpo rígido: rotação em torno do próprio centroide mais
    translação. O bolso fica parado, como num docking real. Devolve as posições
    novas e o RMSD efetivo de cada grafo.

    As ARESTAS não são refeitas, só as distâncias. É deliberado, e não uma
    aproximação por preguiça: manter a topologia fixa deixa `n_contatos` idêntico
    entre a pose nativa e a perturbada, o que fecha o atalho de contagem antes que
    ele exista. O readout entrega `log1p(n_contatos)` direto ao MLP, e sem esse
    cuidado a cabeça de pose aprenderia a contar contatos em vez de ler geometria —
    o mesmo erro que a seção 1 encontrou no readout e que a seção 4 encontrou de
    novo no parse da pose cristalográfica.

    **A magnitude é sorteada por grafo, não fixa.** Com deslocamento fixo de 1,5 Å,
    a loss de ranking caiu de 0,042 para 0,0003 em quatro épocas: separar a nativa
    de uma perturbação sempre do mesmo tamanho é fácil, e uma vez resolvida ela para
    de dar gradiente — restando só a estacionariedade, que sozinha puxa de volta
    para a constante. Sorteando entre 0,3 e 3,0 Å a tarefa cobre a faixa onde o
    déficit real está (rho 0,249 abaixo de 3 Å, contra 0,614 do AutodockVina), e as
    perturbações pequenas continuam difíceis depois que as grandes ficaram fáceis.

    O ângulo escala junto com o deslocamento para que uma perturbação "pequena" seja
    pequena nos dois graus de liberdade ao mesmo tempo.
    """
    lig = (node_type == 0)
    if lig.sum() == 0:
        return pos, torch.zeros(num_graphs, device=pos.device)
    centro, b, p, _ = _centroides_ligante(pos, lig, batch, num_graphs)

    escala = torch.rand(num_graphs, 1, device=pos.device)          # [0,1) por grafo
    desloc = desloc_min + escala * (desloc_max - desloc_min)

    eixo = torch.randn(num_graphs, 3, device=pos.device)
    eixo = eixo / eixo.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sinal = torch.where(torch.rand(num_graphs, 1, device=pos.device) < 0.5, -1.0, 1.0)
    ang = sinal * escala * ang_max

    u, th = eixo[b], ang[b]
    v = p - centro[b]
    cos, sin = torch.cos(th), torch.sin(th)
    # Rodrigues
    v_rot = (v * cos
             + torch.cross(u, v, dim=1) * sin
             + u * (u * v).sum(dim=1, keepdim=True) * (1 - cos))

    t = torch.randn(num_graphs, 3, device=pos.device)
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8) * desloc

    saida = pos.clone()
    saida[lig] = centro[b] + v_rot + t[b]

    # RMSD efetivo por grafo — a rotação contribui além da translação, então ele não
    # é igual a `desloc` e precisa ser medido.
    d2 = (saida[lig] - p).pow(2).sum(dim=1)
    contagem = torch.zeros(num_graphs, device=pos.device).index_add_(
        0, b, torch.ones_like(d2))
    soma = torch.zeros(num_graphs, device=pos.device).index_add_(0, b, d2)
    rmsd = (soma / contagem.clamp(min=1.0)).sqrt()

    return saida, rmsd


def distancias_de(pos, edge_index):
    return torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=1).view(-1, 1)


def forca_e_torque(score, pos, node_type, batch, num_graphs):
    """Força e torque resultantes sobre o ligante, por grafo.

    Se o score vai se comportar como energia, ambos se anulam na pose
    cristalográfica — é a condição de equilíbrio, e o análogo aqui do resíduo de
    EDP de um PINN. Não existe EDP cuja solução seja pKd (é energia livre, com
    entropia e dessolvatação dentro), mas existe esta condição variacional, e ela
    é exatamente o que falta: o déficit medido está na vizinhança do mínimo.

    São impostos os 6 graus de liberdade de CORPO RÍGIDO, não os 3N cartesianos.
    A pose cristalográfica não é o mínimo da geometria interna do ligante segundo o
    modelo, e exigir isso injetaria ruído em vez de sinal.

    Medido antes de qualquer treino, no 1a30: |F| = 1,52 e |T| = 2,01.
    """
    grad = torch.autograd.grad(score.sum(), pos, create_graph=True)[0]
    lig = (node_type == 0)
    centro, b, p, contagem = _centroides_ligante(pos, lig, batch, num_graphs)
    g = grad[lig]

    forca = torch.zeros(num_graphs, 3, device=pos.device).index_add_(0, b, g)
    r = p - centro[b]
    torque = torch.zeros(num_graphs, 3, device=pos.device).index_add_(
        0, b, torch.cross(r, g, dim=1))

    # Normalizadas por número de átomos: sem isso um ligante grande contribuiria
    # mais para a loss por ser grande, não por estar mais longe do equilíbrio.
    return forca / contagem.clamp(min=1.0), torque / contagem.clamp(min=1.0)


def calculate_lj_potential(edge_attr, edge_index, batch, num_graphs,
                           edge_type=None, sigma=3.5):
    """Proxy de potencial estérico de Lennard-Jones, por grafo.

    A versão anterior produzia um alvo CONSTANTE: medido em 400 grafos, o valor
    era 10.000 em 100% deles, desvio-padrão zero. A cabeça auxiliar aprendia a
    prever o número 10, e o termo de física não carregava informação nenhuma.
    Três causas, corrigidas aqui:

    1. **Arestas covalentes entravam no somatório.** Elas têm mediana de 1.38 Å,
       e com σ = 3.0 cada uma contribuía com (3/1.38)^12 ≈ 3·10^4. Em campo de
       força, átomos ligados não interagem por Lennard-Jones — o termo se aplica a
       pares NÃO ligados. Agora só as arestas de interação ligante–proteína entram
       (mediana 4.32 Å, exatamente o regime que o LJ descreve).

    2. **σ = 3.0 era pequeno demais** para contatos C/N/O. O valor típico fica
       entre 3.5 e 4.0 Å; com 3.5, a mediana das distâncias de interação cai no
       poço atrativo em vez da parede repulsiva.

    3. **Somava em vez de promediar.** A soma escala com o tamanho do bolso, então
       o alvo media volume de bolso, não aglomeração. Aqui é a média por aresta.

    A compressão logarítmica preservando o sinal domina a cauda da parede
    repulsiva: sem ela o alvo ia de −0.5 a 395 e o MSE seria governado por poucos
    contatos muito próximos.

    Distribuição resultante (1200 grafos): faixa [−0.43, 6.22], média 1.78,
    desvio-padrão 1.37, correlação com pKd de −0.09. A correlação baixa é
    desejável: um alvo auxiliar muito correlacionado com o principal seria um
    atalho, não um regularizador.
    """
    if edge_type is not None:
        mascara = edge_type >= TIPOS_NAO_LIGADOS
    else:
        # Sem edge_type disponível, separa pela geometria: o raio covalente usado
        # em data_processor.py é 2.0 Å, então tudo acima disso é não-ligado.
        mascara = edge_attr.flatten() > 2.0

    r = torch.clamp(edge_attr[mascara].flatten(), min=1.5)
    term = (sigma / r) ** 6
    lj_energy = term ** 2 - 2 * term

    edge_batch = batch[edge_index[0][mascara]]
    soma = torch.zeros(num_graphs, device=edge_attr.device).scatter_add_(0, edge_batch, lj_energy)
    contagem = torch.zeros(num_graphs, device=edge_attr.device).scatter_add_(
        0, edge_batch, torch.ones_like(lj_energy))
    media = soma / contagem.clamp(min=1.0)

    return (torch.sign(media) * torch.log1p(media.abs())).view(-1, 1)

def train():
    # Nome do checkpoint, sobrescrevível por ambiente: permite rodar um
    # experimento de arquitetura sem destruir o melhor modelo já validado.
    ckpt = os.environ.get("PHARM_CKPT", "pharm_model_weights_best.pth")
    print(f"Checkpoint de saída: {ckpt}")

    # Semente fixa para tornar as comparacoes entre configuracoes pareadas: a
    # mesma semente da a mesma inicializacao e a mesma ordem de lotes, entao a
    # diferenca entre duas rodadas e a configuracao, nao o sorteio.
    seed = int(os.environ.get("PHARM_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Semente: {seed}")

    # Como tratar os 51,3% de IC50 do conjunto de treino, ausentes da validação
    # e do teste. "keep": comportamento historico, todos misturados sem
    # distincao. "drop": descarta o IC50, treino cai para ~6.700 complexos.
    # "token": mantem tudo e informa o tipo de ensaio ao modelo.
    # Restringe o treino a uma lista de pdb_ids (JSON). Usado para o teste de
    # homologia: remover do treino toda proteina parecida com alguma do CASF core
    # responde se o R medido reflete prever um ligante novo num alvo NOVO, ou
    # apenas num alvo ja visto com outro ligante.
    ids_treino = None
    if os.environ.get("PHARM_TREINO_IDS"):
        import json as _json
        ids_treino = set(_json.load(open(os.environ["PHARM_TREINO_IDS"])))
        print(f"Treino restrito a {len(ids_treino)} ids de {os.environ['PHARM_TREINO_IDS']}")

    modo_ic50 = os.environ.get("PHARM_IC50", "keep").lower()
    assert modo_ic50 in ("keep", "drop", "token"), f"PHARM_IC50 invalido: {modo_ic50}"
    print(f"Tratamento do IC50: {modo_ic50}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Treinando em: {device}")

    print("Carregando o Banco de Dados...")
    dataset = PDBbindDataset(root='.')
    
    # =========================================================================
    # CORTE HERMÉTICO: DATA LEAKAGE = ZERO
    # =========================================================================
    print("Separando Treino (General), Validação (Refined) e Teste (CASF core)...")
    refined_set_ids = load_refined_ids()
    core_set_ids = load_core_ids()
    tipo_ensaio = carrega_tipo_ensaio()
    idx_ensaio = {t: i for i, t in enumerate(PharmGeometricGNN.TIPOS_ENSAIO)}

    train_dataset = []
    val_dataset = []
    test_dataset = []

    # A ordem dos testes importa: o core set é subconjunto do refined, então ele
    # precisa ser retirado ANTES, senão os mesmos complexos apareceriam em
    # validação e teste e o teste deixaria de ser independente.
    for graph in dataset:
        if not hasattr(graph, 'pdb_id'):
            continue

        t = tipo_ensaio.get(graph.pdb_id, 'desconhecido')
        graph.assay = torch.tensor([[idx_ensaio[t]]], dtype=torch.long)

        if graph.pdb_id in core_set_ids:
            test_dataset.append(graph)
        elif graph.pdb_id in refined_set_ids:
            val_dataset.append(graph)
        elif modo_ic50 == "drop" and t == "IC50":
            continue   # descartado do treino; nunca entra em val/teste de todo modo
        elif ids_treino is not None and graph.pdb_id not in ids_treino:
            continue   # fora da lista restrita
        else:
            train_dataset.append(graph)

    total_grafos = len(dataset)
    print(f"Total: {total_grafos} | Treino: {len(train_dataset)} | "
          f"Validação: {len(val_dataset)} | Teste (CASF): {len(test_dataset)}")

    # `InMemoryDataset.get()` guarda uma cópia rasa de cada grafo acessado em
    # `_data_list`, então a varredura acima deixa os 19.037 em cache. As três
    # listas já seguram tudo o que o treino precisa — os tensores são views
    # (`narrow`) sobre o tensor colado, não cópias — e o cache passa a ser um
    # segundo conjunto de objetos Python sem uso. Medido: ~70 MB nos 19 mil.
    # Pouco, mas é memória parada durante todas as épocas, e zerá-la é de graça.
    dataset._data_list = None
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) if test_dataset else None

    # Inicializar o Modelo (Dropout ativo = 0.2 default)
    # Física de pose: desligada por padrão, para que `python train.py` continue
    # reproduzindo exatamente a linha de referência de R = 0,754.
    usar_pose = os.environ.get("PHARM_POSE", "0") not in ("0", "", "off", "no")
    # Camadas com features vetoriais equivariantes (ver CamadaDirecional em
    # gnn_model.py). Desligado por padrão: muda o state_dict por completo, e os
    # checkpoints existentes são de camadas escalares.
    usar_direcional = os.environ.get("PHARM_DIRECIONAL", "0") not in ("0", "", "off", "no")
    lambda_rank = float(os.environ.get("PHARM_LAMBDA_RANK", "1.0"))
    lambda_esta = float(os.environ.get("PHARM_LAMBDA_ESTA", "0.1"))
    margem_rank = float(os.environ.get("PHARM_MARGEM", "0.5"))  # por Å de RMSD
    # Épocas antes de ligar a estacionariedade. Medido na inicialização: a cabeça de
    # pose nasce praticamente constante (|F| = 1e-4), e uma constante satisfaz a
    # estacionariedade PERFEITAMENTE sem ter aprendido nada. Ligada desde a época 1
    # ela competiria com o ranking justamente enquanto este tenta tirar a cabeça da
    # constante. O ranking dá conteúdo ao mínimo; a estacionariedade depois lhe dá
    # forma. Ordem inversa arrisca travar na solução trivial.
    esta_warmup = int(os.environ.get("PHARM_ESTA_WARMUP", "5"))

    model = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4,
                              usar_assay=(modo_ic50 == "token"),
                              usar_pose=usar_pose,
                              usar_direcional=usar_direcional).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[*] Modelo com {n_par:,} parâmetros"
          + (" · camadas DIRECIONAIS (distância + ângulo)" if usar_direcional
             else " · camadas escalares (só distância)"))
    if usar_pose:
        print(f"[*] Física de pose ATIVA — ranking (peso {lambda_rank}, margem "
              f"{margem_rank}/Å de RMSD, perturbação 0,3–3,0 Å) + estacionariedade "
              f"(peso {lambda_esta}, a partir da época {esta_warmup + 1}).")
    
    # Regularização L2: weight_decay penaliza pesos gigantes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion = nn.MSELoss()
    lambda_physics = 0.1 # Peso da restrição física na Loss global

    epochs = 300

    # Paciência do early stopping. Precisa ser maior que a do ReduceLROnPlateau
    # (10), senão o treino pararia antes de a redução da taxa ter chance de
    # ajudar. Com 30, cabem duas reduções de LR antes da desistência.
    #
    # Na rodada anterior isso teria encerrado por volta da época 54: o melhor
    # modelo saiu na época 24 e a validação só piorou depois, de 2.27 para 3.07,
    # enquanto o treino caía para 0.365. As 276 épocas restantes foram gastas com
    # a taxa de aprendizado já zerada e nenhuma melhora possível.
    paciencia_parada = 30
    print(f"Iniciando o treinamento por até {epochs} épocas "
          f"(parada antecipada após {paciencia_parada} sem melhora)...")

    best_val_loss = float('inf')
    melhor_epoca = -1
    epocas_sem_melhora = 0

    for epoch in range(epochs):
        model.train()
        total_loss_pkd = 0
        total_loss_physics = 0
        total_loss_rank = 0
        total_loss_esta = 0
        lotes_descartados = 0
        
        for data in train_loader:
            data_homo = data.to_homogeneous().to(device)
            target_pkd = data.y.to(device)
            assay = data.assay.to(device)

            # Gerar target de física dinamicamente
            target_steric = calculate_lj_potential(
                data_homo.edge_attr, data_homo.edge_index, data_homo.batch,
                data.num_graphs, edge_type=data_homo.edge_type
            )

            optimizer.zero_grad()

            if not usar_pose:
                out_pkd, out_steric = model(data_homo, assay=assay)
                loss_pkd = criterion(out_pkd, target_pkd)
                loss_physics = criterion(out_steric, target_steric)
                loss = loss_pkd + lambda_physics * loss_physics
            else:
                # `pos` precisa de gradiente para a estacionariedade. O forward já
                # recomputa `edge_attr` a partir de `pos` quando isso acontece.
                pos_nativa = data_homo.pos.detach().clone().requires_grad_(True)
                data_homo.pos = pos_nativa

                out_pkd, out_steric, out_pose = model(
                    data_homo, assay=assay, retornar_pose=True)
                loss_pkd = criterion(out_pkd, target_pkd)
                loss_physics = criterion(out_steric, target_steric)

                # (a) Estacionariedade: na pose cristalográfica, força e torque
                #     resultantes sobre o ligante devem se anular.
                forca, torque = forca_e_torque(
                    out_pose, pos_nativa, data_homo.node_type,
                    data_homo.batch, data.num_graphs)
                loss_esta = (forca.pow(2).sum(dim=1) + torque.pow(2).sum(dim=1)).mean()

                # (b) Ranking contra uma perturbação rígida. Sem ele a
                #     estacionariedade é DEGENERADA: uma cabeça constante zera
                #     força e torque perfeitamente sem ter aprendido nada. O
                #     ranking é o que dá conteúdo ao mínimo; a estacionariedade é o
                #     que lhe dá forma.
                with torch.no_grad():
                    pos_pert, rmsd_pert = perturbar_ligante(
                        pos_nativa.detach(), data_homo.node_type,
                        data_homo.batch, data.num_graphs)
                    attr_pert = distancias_de(pos_pert, data_homo.edge_index)

                attr_original = data_homo.edge_attr
                data_homo.pos = pos_pert          # sem requires_grad: não há
                data_homo.edge_attr = attr_pert   # estacionariedade a impor aqui
                _, _, pose_pert = model(data_homo, assay=assay, retornar_pose=True)
                data_homo.edge_attr = attr_original

                # Margem PROPORCIONAL ao afastamento: uma perturbação de 0,3 Å
                # exige pouca diferença de score, uma de 3 Å exige muita. Com
                # margem fixa a cabeça só precisa aprender o SINAL da diferença,
                # e satura; proporcional, ela precisa aprender o quanto o score
                # decai com a distância — que é a estrutura fina que falta.
                margem = margem_rank * rmsd_pert.view(-1, 1)
                loss_rank = torch.relu(
                    margem - (out_pose - pose_pert)).mean()

                peso_esta = lambda_esta if epoch >= esta_warmup else 0.0
                loss = (loss_pkd + lambda_physics * loss_physics
                        + lambda_rank * loss_rank + peso_esta * loss_esta)

            if not torch.isfinite(loss):
                # Um lote patológico não deve contaminar os pesos. Descartar é
                # melhor que propagar: o NaN é absorvente, e uma vez nos pesos
                # todas as épocas seguintes viram NaN.
                lotes_descartados += 1
                optimizer.zero_grad()
                continue

            loss.backward()

            # Recorte de norma do gradiente — só no caminho direcional.
            #
            # Com double backward sobre features vetoriais a distribuição de norma
            # do gradiente tem cauda pesada, e um único lote basta para levar os
            # pesos a uma região de onde não voltam: treino saudável até a época 20
            # (Estac ~0,007), explosão na 21 (Estac 813, pKd 523), NaN na 22.
            #
            # O limite de 500 vem de medição, não de convenção. As normas em treino
            # saudável, sobre 20 lotes:
            #
            #     escalar + física     mediana 53,6 · p95 105,7 · máx 149,0
            #     direcional + física  mediana 54,1 · p95 227,0 · máx 325,3
            #
            # Um limite "seguro" de 10, que é o default folclórico, cortaria 100%
            # dos lotes e mudaria o treino por completo — inclusive o escalar, cujo
            # resultado já está reportado. 500 fica acima de todo o regime normal e
            # só corta a explosão.
            #
            # Restrito a `usar_direcional` de propósito: o caminho escalar chegou ao
            # R = 0,750 e aos 79,3% de top1 sem recorte nenhum, e mudá-lo agora
            # tornaria aqueles números irreprodutíveis.
            # A norma é sempre calculada, mesmo sem recorte, porque é ela que
            # denuncia um gradiente podre. `clip_grad_norm_` devolve a norma ANTES
            # do recorte, então serve às duas coisas.
            norma = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=(500.0 if usar_direcional else float('inf')))

            # Checar a LOSS não basta, e a rodada anterior mostrou por quê: uma loss
            # finita pode ter gradiente NaN — `sqrt(0)` é o caso clássico, valor
            # finito e derivada infinita. Um único lote assim injeta NaN nos pesos,
            # e a partir daí TODA loss vira NaN: nas épocas 11 a 19 caía 1 lote por
            # época, e na 20 todos os 429 caíram de uma vez, com a validação
            # congelada. O descarte passa a proteger um modelo já morto.
            if not torch.isfinite(norma):
                lotes_descartados += 1
                optimizer.zero_grad()
                continue

            optimizer.step()
            
            total_loss_pkd += loss_pkd.item() * data.num_graphs
            total_loss_physics += loss_physics.item() * data.num_graphs
            if usar_pose:
                total_loss_rank += loss_rank.item() * data.num_graphs
                total_loss_esta += loss_esta.item() * data.num_graphs
            
        avg_train_loss_pkd = total_loss_pkd / len(train_dataset)
        avg_train_loss_phys = total_loss_physics / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss_pkd = 0
        with torch.no_grad():
            for data in val_loader:
                data_homo = data.to_homogeneous().to(device)
                target_pkd = data.y.to(device)
                # Validação e teste sao 100% Kd/Ki; o token de ensaio vem do proprio dado.
                out_pkd, _ = model(data_homo, assay=data.assay.to(device))
                val_loss_pkd += criterion(out_pkd, target_pkd).item() * data.num_graphs
                
        avg_val_loss_pkd = val_loss_pkd / len(val_dataset)
        
        # O Scheduler escuta estritamente a Loss de pKd da Validação
        scheduler.step(avg_val_loss_pkd)
        
        current_lr = optimizer.param_groups[0]['lr']
        extra = ""
        if usar_pose:
            # |F| e |T| caindo ao longo das épocas é a métrica de progresso da
            # estacionariedade: mede o quanto a pose cristalográfica virou um
            # equilíbrio para a cabeça de pose.
            extra = (f" | Rank: {total_loss_rank / len(train_dataset):.4f}"
                     f" | Estac: {total_loss_esta / len(train_dataset):.5f}")
        if lotes_descartados:
            extra += f" | descartados: {lotes_descartados}"
        print(f"Época [{epoch+1:03d}/{epochs:03d}] | Train pKd: {avg_train_loss_pkd:.4f} | Train Phys: {avg_train_loss_phys:.4f}{extra} | Val pKd: {avg_val_loss_pkd:.4f} | LR: {current_lr:.6f}")
        
        # =========================================================================
        # EARLY STOPPING E CHECKPOINTING DE MELHOR MODELO
        # =========================================================================
        if avg_val_loss_pkd < best_val_loss:
            best_val_loss = avg_val_loss_pkd
            melhor_epoca = epoch + 1
            epocas_sem_melhora = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Novo melhor modelo salvo! (Val pKd: {best_val_loss:.4f})")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia_parada:
                print(f"\n[!] Parada antecipada na época {epoch+1}: "
                      f"{paciencia_parada} épocas sem melhora na validação.")
                print(f"    Melhor modelo: época {melhor_epoca}, Val pKd {best_val_loss:.4f}")
                break

    print(f"\nTreinamento finalizado. Melhor Val pKd: {best_val_loss:.4f} "
          f"(época {melhor_epoca}) | RMSE de validação: {best_val_loss ** 0.5:.3f}")

    # =========================================================================
    # AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
    # Executada UMA vez, com o checkpoint escolhido pela validação. Este é o
    # número reportável: o core set não participou do treino nem da seleção.
    # =========================================================================
    if test_loader is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        preds, alvos = [], []
        with torch.no_grad():
            for data in test_loader:
                out, _ = model(data.to_homogeneous().to(device), assay=data.assay.to(device))
                preds.append(out.squeeze(-1).cpu())
                alvos.append(data.y.squeeze(-1).cpu())
        p = torch.cat(preds).numpy()
        t = torch.cat(alvos).numpy()

        rmse = float(((p - t) ** 2).mean() ** 0.5)
        mae = float(abs(p - t).mean())
        r = float(np.corrcoef(p, t)[0, 1])

        print("\n" + "=" * 60)
        print(f" TESTE — CASF-2016 core set ({len(t)} complexos)")
        print("=" * 60)
        print(f"  RMSE      : {rmse:.3f} unidades de pKd")
        print(f"  MAE       : {mae:.3f}")
        print(f"  Pearson R : {r:.3f}   (R² = {r**2:.3f})")
        print(f"  Baseline (prever a média): RMSE {float(t.std()):.3f}")

if __name__ == "__main__":
    train()
