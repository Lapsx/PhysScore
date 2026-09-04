import torch
from torch_geometric.data import InMemoryDataset
import gc
import os
import re
import math
import time
from data_processor import PharmGraphBuilder
from pharmacophore_labels import rotular_ligante


def _gravar_parte(diretorio, grafos):
    """Grava um bloco de grafos com nome derivado do conteúdo.

    O nome vem do primeiro e do último pdb_id do bloco, não de um contador: se a
    execução cair e recomeçar, o mesmo bloco regrava o mesmo arquivo em vez de
    criar duplicata.
    """
    if not grafos:
        return
    ids = [g.pdb_id for g in grafos if hasattr(g, "pdb_id")]
    nome = f"{ids[0]}_{ids[-1]}_{len(grafos)}.pt" if ids else f"bloco_{len(grafos)}.pt"
    torch.save(grafos, os.path.join(diretorio, nome))


def _rss_mb():
    """Memória residente do processo, em MB. Devolve None fora do Linux."""
    try:
        with open('/proc/self/status') as f:
            for linha in f:
                if linha.startswith('VmRSS'):
                    return int(linha.split()[1]) / 1024.0
    except OSError:
        pass
    return None

class PDBbindDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Como as subpastas podem variar (19.000 IDs), não listamos todas aqui rigidamente
        return []

    @property
    def processed_file_names(self):
        return ['pharm_xai_dataset.pt']

    def parse_labels(self):
        """Lê o arquivo INDEX_general_PL.2020R1.lst e converte as afinidades para -log(Kd/Ki)"""
        labels = {}
        index_file = os.path.join(self.root, 'index', 'INDEX_general_PL.2020R1.lst')
        
        if not os.path.exists(index_file):
            print(f"AVISO: Arquivo de rótulos não encontrado em {index_file}")
            return labels
            
        print("Mapeando Afinidades Termodinâmicas experimentais do arquivo INDEX...")
        with open(index_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 4: continue
                
                pdb_id = parts[0]
                binding_data = parts[3] # Ex: Kd=49uM ou Ki=0.35nM
                
                # Regex para extrair valor e unidade
                match = re.search(r'(=|~|<|>)\s*([\d\.]+)\s*(mM|uM|nM|pM|fM)', binding_data)
                if match:
                    val = float(match.group(2))
                    unit = match.group(3)
                    
                    # Convertendo tudo para Molar (M) absoluto
                    if unit == 'mM': val *= 1e-3
                    elif unit == 'uM': val *= 1e-6
                    elif unit == 'nM': val *= 1e-9
                    elif unit == 'pM': val *= 1e-12
                    elif unit == 'fM': val *= 1e-15
                    
                    if val > 0:
                        # pKd = -log10(Kd)
                        pKd = -math.log10(val)
                        labels[pdb_id] = pKd
        
        print(f"Rótulos (-logK) mapeados para {len(labels)} complexos.")
        return labels

    def process(self):
        print("Iniciando o processamento dos arquivos brutos (.pdb/.sdf) em lote...")
        data_list = []
        builder = PharmGraphBuilder(pocket_radius=6.0)
        
        # 1. Carregar o Dicionário de Afinidades (Ground Truth / Labels)
        affinity_dict = self.parse_labels()

        # Ordenado de propósito: `os.listdir` devolve ordem de inode, que muda entre
        # máquinas e entre execuções. Com a lista ordenada, o "complexo N" de um log
        # é o mesmo complexo numa nova execução, e o contador abaixo vira um
        # diagnóstico utilizável — sem isso, saber onde a execução anterior parou não
        # ajuda a reproduzir. A ordem não afeta o resultado: o split é por ID e o
        # DataLoader embaralha.
        folders = sorted(f for f in os.listdir(self.raw_dir)
                         if os.path.isdir(os.path.join(self.raw_dir, f)))
        total = len(folders)
        print(f"Encontrados {total} complexos na pasta raw/. Processando...")

        # Este laço leva horas e era completamente silencioso. Quando a máquina
        # travou durante ele, não havia como saber em que ponto — nem se o problema
        # crescia aos poucos ou era um complexo específico. O RSS na linha responde
        # às duas perguntas de uma vez: se sobe monotonicamente é acúmulo, se pula
        # de repente é o complexo nomeado em `atual`.
        inicio = time.time()
        passo = 250

        # ── Retomada ────────────────────────────────────────────────────────
        # O `process()` só gravava no fim. Numa queda de energia a 66% do
        # caminho, 8h45 de trabalho evaporam porque nada tinha ido a disco.
        # Agora os grafos são despejados em partes a cada `passo_parcial`, e uma
        # execução nova reaproveita o que já existe. O custo é irrisório perto do
        # de reconstruir: cada parte tem ~250 grafos e leva menos de um segundo
        # para gravar.
        dir_parcial = os.path.join(self.processed_dir, "parciais")
        os.makedirs(dir_parcial, exist_ok=True)
        passo_parcial = 500

        feitos = set()
        for arq in sorted(os.listdir(dir_parcial)):
            if not arq.endswith(".pt"):
                continue
            try:
                bloco = torch.load(os.path.join(dir_parcial, arq), weights_only=False)
            except Exception:
                print(f"  [!] parte corrompida, ignorando: {arq}")
                continue
            # Só os IDs interessam aqui. Manter os grafos vivos custaria caro —
            # ver a nota sobre custo por população viva mais abaixo.
            feitos.update(g.pdb_id for g in bloco if hasattr(g, "pdb_id"))
            del bloco
        gc.collect()
        if feitos:
            print(f"[+] Retomando: {len(feitos)} complexos já processados em "
                  f"{dir_parcial}. Apague essa pasta para começar do zero.")

        pendentes = []
        n_feitos = 0

        for i, folder in enumerate(folders, start=1):
            if i % passo == 0 or i == total:
                decorrido = time.time() - inicio
                restante = decorrido / i * (total - i)
                rss = _rss_mb()
                rss_txt = f"RSS {rss:6.0f} MB | " if rss is not None else ""
                print(f"  [{i:5d}/{total}] {len(feitos) + n_feitos:5d} grafos | {rss_txt}"
                      f"{decorrido/60:5.1f} min decorridos, ~{restante/60:.0f} min restantes "
                      f"| atual: {folder}", flush=True)

            pdb_id = folder.lower()
            if pdb_id not in affinity_dict:
                continue # Ignoramos complexos sem rótulo experimental
            if pdb_id in feitos:
                continue # já veio de uma parte gravada
                
            folder_path = os.path.join(self.raw_dir, folder)
            protein_path = os.path.join(folder_path, f"{pdb_id}_protein.pdb")
            
            # PDBbind costuma usar SDF, MOL2 ou PDB
            ligand_sdf = os.path.join(folder_path, f"{pdb_id}_ligand.sdf")
            ligand_mol2 = os.path.join(folder_path, f"{pdb_id}_ligand.mol2")
            ligand_pdb = os.path.join(folder_path, f"ligand.pdb") 
            
            if os.path.exists(ligand_sdf): ligand_path = ligand_sdf
            elif os.path.exists(ligand_mol2): ligand_path = ligand_mol2
            else: ligand_path = ligand_pdb

            if os.path.exists(protein_path) and os.path.exists(ligand_path):
                try:
                    # 2. Constrói a Física do Grafo
                    graph = builder.build_hetero_graph(protein_path, ligand_path)
                    
                    # 3. Adiciona a Resposta Certa (Loss Target)
                    graph.y = torch.tensor([[affinity_dict[pdb_id]]], dtype=torch.float)
                    
                    # Salva o ID (crachá) para podermos fazer o corte Refined vs General depois!
                    graph.pdb_id = pdb_id

                    # Rótulos farmacofóricos por átomo do ligante, (n_lig, 6).
                    # Anexados aqui porque derivá-los custa uma cdist sobre um
                    # grafo que já está montado — e porque reprocessar os 19 mil
                    # complexos leva horas. Uma passagem serve tanto ao modelo de
                    # afinidade quanto ao de farmacóforo.
                    graph.y_pharm = rotular_ligante(graph)
                    
                    # `data_list` NÃO recebe o grafo aqui. Medido em 60 complexos
                    # reprocessados sob populações vivas crescentes: 1,46 s/complexo
                    # com a lista vazia, 1,79 s com 2.000 grafos vivos, 2,17 s com
                    # 4.000 e 2,51 s com 8.000 — +72%, monotônico. Não é o coletor de
                    # lixo (medido à parte: 39,6 ms contra 49,0 ms de coleta) nem I/O
                    # (0,3 MB/s, processo em estado R a 100% de CPU): é o custo de
                    # alocar sobre um heap com dezenas de milhares de tensores vivos.
                    # Como cada grafo já vai a disco em blocos de `passo_parcial`, a
                    # lista era pura despesa — ela só é reconstruída no fim, para o
                    # `collate`, quando não há mais nada a processar.
                    n_feitos += 1
                    pendentes.append(graph)
                    if len(pendentes) >= passo_parcial:
                        _gravar_parte(dir_parcial, pendentes)
                        pendentes = []
                        graph = None
                        gc.collect()
                except Exception as e:
                    pass

        if pendentes:
            _gravar_parte(dir_parcial, pendentes)
            pendentes = []
            graph = None
            gc.collect()

        # Só agora a lista completa é montada, lendo de volta o que foi gravado.
        # Nada mais será processado depois deste ponto, então o custo de manter
        # 19 mil grafos vivos não recai sobre nenhuma construção de grafo.
        print("\nLendo as partes gravadas para montar o dataset final...")
        for arq in sorted(os.listdir(dir_parcial)):
            if not arq.endswith(".pt"):
                continue
            try:
                bloco = torch.load(os.path.join(dir_parcial, arq), weights_only=False)
            except Exception:
                print(f"  [!] parte corrompida, ignorando: {arq}")
                continue
            data_list.extend(bloco)
            del bloco

        print(f"\n{len(data_list)} grafos rotulados com sucesso!")
        n_grafos = len(data_list)

        # `collate` monta uma segunda cópia integral do dataset enquanto os grafos
        # individuais ainda estão vivos, e `torch.save` serializa uma terceira. Com
        # os 19.037 complexos do PDBbind o pico chegava a ~3x o dataset, e numa
        # máquina de 15 GB com 2 GB de swap esse pico — que acontece justamente no
        # FIM de um processamento de várias horas — era suficiente para levar a
        # máquina a thrashing. Soltar `data_list` entre as duas etapas derruba o
        # pico de 3x para 2x, e é de graça: a lista não é mais usada depois daqui.
        data, slices = self.collate(data_list)
        del data_list
        graph = None   # a última volta do laço ainda segura um grafo pelo nome
        gc.collect()

        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset salvo em: {self.processed_paths[0]} ({n_grafos} grafos)")

        # As partes já cumpriram o papel; o dataset colado é a fonte definitiva.
        import shutil
        shutil.rmtree(dir_parcial, ignore_errors=True)
        print(f"[+] Partes intermediárias removidas.")

if __name__ == "__main__":
    dataset = PDBbindDataset(root='.')
    print("\nResumo:")
    print(f"Tamanho total: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Amostra [0]: {dataset[0]}")
        print(f"Target de Afinidade (data.y) [0]: {dataset[0].y.item():.3f}")
