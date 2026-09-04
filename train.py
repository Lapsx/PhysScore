import torch
import numpy as np
import gc
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import re

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

    # Como tratar os 51,3% de IC50 do conjunto de treino, ausentes da validação
    # e do teste. "keep": comportamento historico, todos misturados sem
    # distincao. "drop": descarta o IC50, treino cai para ~6.700 complexos.
    # "token": mantem tudo e informa o tipo de ensaio ao modelo.
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
    model = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4,
                              usar_assay=(modo_ic50 == "token")).to(device)
    
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
        
        for data in train_loader:
            data_homo = data.to_homogeneous().to(device)
            target_pkd = data.y.to(device)
            
            # Gerar target de física dinamicamente
            target_steric = calculate_lj_potential(
                data_homo.edge_attr, data_homo.edge_index, data_homo.batch,
                data.num_graphs, edge_type=data_homo.edge_type
            )
            
            optimizer.zero_grad()
            out_pkd, out_steric = model(data_homo, assay=data.assay.to(device))
            
            loss_pkd = criterion(out_pkd, target_pkd)
            loss_physics = criterion(out_steric, target_steric)
            
            # Loss Multi-Tarefa
            loss = loss_pkd + lambda_physics * loss_physics
            
            loss.backward()
            optimizer.step()
            
            total_loss_pkd += loss_pkd.item() * data.num_graphs
            total_loss_physics += loss_physics.item() * data.num_graphs
            
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
        print(f"Época [{epoch+1:03d}/{epochs:03d}] | Train pKd: {avg_train_loss_pkd:.4f} | Train Phys: {avg_train_loss_phys:.4f} | Val pKd: {avg_val_loss_pkd:.4f} | LR: {current_lr:.6f}")
        
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
