"""
Mapa de consenso do sítio de ligação por cluster do CASF-2016
==============================================================
Agrega a saliência do modelo sobre os CINCO ligantes quimicamente distintos que
se ligam ao mesmo alvo, identificando os resíduos do bolso dos quais a predição
depende de forma consistente.

Por que isto é diferente de `explain.py`
----------------------------------------
`explain.py` produz um mapa de saliência de UM complexo: para aquele par
proteína–ligante específico, quais distâncias mais influenciam a afinidade
predita. É informação sobre o modelo naquele ponto, e não é transferível.

Um farmacóforo, em química medicinal, é outra coisa: uma descrição das
características necessárias ao reconhecimento, derivada de um CONJUNTO de
moléculas ativas e descrevendo o que elas têm em comum. O elemento que falta na
saliência de um complexo é justamente o consenso.

O CASF-2016 fornece exatamente o dado que preenche essa lacuna: 57 clusters de
cinco complexos cada, todos do mesmo alvo, com ligantes quimicamente distintos.
Um resíduo que aparece entre os mais salientes para os cinco ligantes é candidato
legítimo a ponto quente de reconhecimento — porque a dependência sobrevive à troca
completa da química do ligante.

Ressalvas que o texto e o README devem carregar
------------------------------------------------
1. Isto é um mapa do lado do RECEPTOR (resíduos), não um farmacóforo do lado do
   ligante (grupos funcionais com geometria). É o complemento, não o mesmo objeto.
2. O consenso é sobre o que o MODELO usa, não necessariamente sobre o que a
   natureza usa. Concordância com interações conhecidas é evidência a favor do
   modelo; discordância é evidência de atalho aprendido.
3. Cinco ligantes é uma amostra pequena para consenso.

Uso
---
    python3 consenso_farmacoforo.py 1              # cluster 1
    python3 consenso_farmacoforo.py --pdb 3uuo     # o cluster que contém 3uuo
    python3 consenso_farmacoforo.py 1 --top 8      # quantos resíduos destacar
"""

import argparse
import collections
import os
import sys

import numpy as np
import torch

from data_processor import PharmGraphBuilder
from gnn_model import PharmGeometricGNN

PESOS = "pharm_model_weights_best.pth"
CORE = "core_set.dat"


def ler_clusters(caminho=CORE):
    """cluster -> [pdb_ids] e pdb_id -> cluster, a partir do CoreSet.dat."""
    por_cluster = collections.defaultdict(list)
    de_pdb = {}
    if not os.path.exists(caminho):
        print(f"[!] {caminho} não encontrado. Extraia CoreSet.dat do pacote CASF-2016.")
        sys.exit(1)
    for linha in open(caminho):
        if linha.startswith('#'):
            continue
        p = linha.split()
        if len(p) >= 6:
            por_cluster[p[5]].append(p[0].lower())
            de_pdb[p[0].lower()] = p[5]
    return por_cluster, de_pdb


# Resíduos excluídos do consenso por não terem identidade estável entre entradas.
#
# Água cristalográfica ocupa cerca de 30% do bolso nos complexos inspecionados, mas
# a numeração de HOH é atribuída por estrutura: HOH147 numa entrada não é a mesma
# molécula que HOH147 em outra. Elas nunca podem formar consenso por número de
# resíduo — só diluem o ranking. Íons têm o mesmo problema de numeração.
#
# Isso NÃO significa que a água seja irrelevante para o reconhecimento: águas-ponte
# conservadas são reais e importantes. Significa que este método, que casa resíduos
# por numeração, não é a ferramenta para encontrá-las — precisaria de superposição
# estrutural e casamento espacial.
IGNORADOS = {'HOH', 'WAT', 'DOD', 'NA', 'CL', 'K', 'MG', 'ZN', 'CA', 'MN', 'NI',
             'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'DMS'}


def saliencia_por_residuo(modelo, builder, pdb_id, raiz='raw', incluir_solvente=False):
    """Saliência do modelo somada por resíduo do bolso, normalizada.

    A saliência de cada aresta é |∂(pKd)/∂(distância)|. Só as arestas de
    INTERAÇÃO entram: as covalentes descrevem a geometria interna das moléculas,
    não o reconhecimento entre elas.

    A normalização (divisão pelo total) é necessária porque bolsos maiores têm
    mais arestas e somariam mais saliência sem que isso signifique dependência
    mais forte — sem ela, o consenso mediria tamanho de bolso.
    """
    prot = f"{raiz}/{pdb_id}/{pdb_id}_protein.pdb"
    lig = f"{raiz}/{pdb_id}/{pdb_id}_ligand.sdf"
    if not (os.path.exists(prot) and os.path.exists(lig)):
        return None

    dados = builder.build_hetero_graph(prot, lig)
    homo = dados.to_homogeneous()

    homo.edge_attr.requires_grad_(True)
    modelo.zero_grad()
    pkd, _ = modelo(homo)
    pkd.sum().backward()
    sal = homo.edge_attr.grad.abs().flatten()

    # Índices do bolso e chaves de resíduo, pela MESMA seleção do grafo
    lig_pos, _ = builder.parse_ligand(lig)
    prot_pos, _ = builder.parse_protein(prot)
    idx_bolso = builder.selecionar_bolso(lig_pos, prot_pos)
    chaves_todas = builder.residuos_proteina(prot)
    chaves_bolso = [chaves_todas[i] for i in idx_bolso]

    n_lig = dados['ligand'].x.shape[0]
    tipos = homo.edge_type
    linha, coluna = homo.edge_index

    # Arestas de interação: tipo >= 2 (0 e 1 são covalentes)
    por_residuo = collections.defaultdict(float)
    for e in torch.nonzero(tipos >= 2).flatten().tolist():
        # o nó de proteína é o que tem índice >= n_lig no grafo homogêneo
        for no in (int(linha[e]), int(coluna[e])):
            if no >= n_lig:
                j = no - n_lig
                if 0 <= j < len(chaves_bolso):
                    chave = chaves_bolso[j]
                    if incluir_solvente or chave[1] not in IGNORADOS:
                        por_residuo[chave] += float(sal[e])
                break

    total = sum(por_residuo.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in por_residuo.items()}


def main():
    ap = argparse.ArgumentParser(description="Consenso de sítio por cluster CASF-2016")
    ap.add_argument("cluster", nargs="?", help="id do cluster (1 a 57)")
    ap.add_argument("--pdb", help="usa o cluster que contém este PDB id")
    ap.add_argument("--top", type=int, default=6,
                    help="resíduos salientes considerados por complexo")
    ap.add_argument("--fracao", type=float, default=None,
                    help="alternativa a --top: fração do bolso de cada complexo")
    ap.add_argument("--raiz", default="raw")
    ap.add_argument("--incluir-solvente", action="store_true",
                    help="mantém água e íons (ver nota sobre numeração instável)")
    args = ap.parse_args()

    por_cluster, de_pdb = ler_clusters()

    if args.pdb:
        alvo = de_pdb.get(args.pdb.lower())
        if alvo is None:
            print(f"[!] {args.pdb} não está no core set.")
            sys.exit(1)
    elif args.cluster:
        alvo = args.cluster
    else:
        ap.error("informe o cluster ou --pdb")

    complexos = por_cluster.get(alvo)
    if not complexos:
        print(f"[!] cluster {alvo} não encontrado.")
        sys.exit(1)

    print(f"[*] Cluster {alvo}: {', '.join(complexos)}")

    modelo = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4)
    modelo.load_state_dict(torch.load(PESOS, map_location='cpu'))
    modelo.eval()
    builder = PharmGraphBuilder(pocket_radius=6.0, interaction_radius=5.0, covalent_radius=2.0)

    mapas, usados = [], []
    for pid in complexos:
        try:
            m = saliencia_por_residuo(modelo, builder, pid, args.raiz,
                                      incluir_solvente=args.incluir_solvente)
        except Exception as e:
            print(f"    {pid}: falhou ({type(e).__name__})")
            continue
        if m:
            mapas.append(m)
            usados.append(pid)
            print(f"    {pid}: {len(m)} resíduos no bolso")

    if len(mapas) < 2:
        print("[!] menos de dois complexos processados: não há consenso a extrair.")
        sys.exit(1)

    # Agregação: média da saliência normalizada e frequência no top-N
    soma = collections.defaultdict(float)
    freq = collections.Counter()
    cortes = []
    for m in mapas:
        for k, v in m.items():
            soma[k] += v
        corte = max(4, int(round(args.fracao * len(m)))) if args.fracao else args.top
        cortes.append(corte)
        for k, _ in sorted(m.items(), key=lambda kv: -kv[1])[:corte]:
            freq[k] += 1
    corte_medio = sum(cortes) / len(cortes)

    n = len(mapas)

    # Diagnóstico da premissa do método: os cinco complexos precisam compartilhar
    # a numeração de resíduos para que o consenso signifique alguma coisa. Medido
    # em sete clusters, sobreposição de 61 a 79% produz consenso 5/5, enquanto
    # sobreposição nula produz no máximo 3/5 — não por rigidez do corte, mas
    # porque não há resíduo comum a comparar. Quando isso acontece, ou os ligantes
    # ocupam sub-sítios realmente distintos, ou as entradas usam convenções de
    # numeração incompatíveis. Nos dois casos o resultado não é interpretável.
    conjuntos = [set(m) for m in mapas]
    uniao = set().union(*conjuntos)
    intersec = set.intersection(*conjuntos)
    sobrep = len(intersec) / max(len(uniao), 1)
    print(f"    resíduos: {sum(len(m) for m in mapas)/n:.1f} por complexo, "
          f"{len(uniao)} distintos ao todo, sobreposição {sobrep*100:.0f}%")
    if sobrep < 0.20:
        print()
        print("[!] Sobreposição muito baixa: os cinco complexos quase não compartilham")
        print("    resíduos de bolso pela numeração. O consenso abaixo NÃO é confiável —")
        print("    ou os ligantes ocupam sub-sítios distintos, ou as entradas usam")
        print("    numerações incompatíveis. Confira as estruturas antes de interpretar.")

    linhas = [(freq[k], soma[k] / n, k) for k in soma]
    # Ordena por consenso primeiro, intensidade depois: um resíduo forte em um só
    # complexo é idiossincrasia daquele ligante, não característica do sítio.
    linhas.sort(key=lambda t: (-t[0], -t[1]))

    print()
    print(f"{'resíduo':>12} | {'consenso':>10} | {'saliência média':>15}")
    print("-" * 44)
    for f, s, (num, nome) in linhas[:15]:
        marca = "  <--" if f == n else ""
        print(f"{nome:>5}{num:<7} | {f}/{n:<8} | {s:15.4f}{marca}")

    unanimes = [(num, nome) for f, s, (num, nome) in linhas if f == n]
    print()
    print(f"[+] {len(unanimes)} resíduos salientes em TODOS os {n} ligantes "
          f"(corte médio: {corte_medio:.0f} de {len(soma)} resíduos do bolso):")
    print(f"    {', '.join(f'{nm}{nu}' for nu, nm in unanimes) if unanimes else '(nenhum)'}")
    print()
    print("    Estes são os candidatos a ponto quente do sítio: a dependência do")
    print("    modelo sobrevive à troca completa da química do ligante.")

    # Macro PyMOL sobre o primeiro complexo do cluster
    ref = usados[0]
    pml = f"consenso_cluster{alvo}.pml"
    with open(pml, 'w') as f:
        f.write(f"# Consenso do cluster {alvo} ({', '.join(usados)})\n")
        f.write(f"load {args.raiz}/{ref}/{ref}_protein.pdb, receptor\n")
        f.write(f"load {args.raiz}/{ref}/{ref}_ligand.sdf, ligante\n")
        f.write("hide everything\nshow cartoon, receptor\ncolor grey80, receptor\n")
        f.write("show sticks, ligante\ncolor cyan, ligante\n")
        for f_, s_, (num, nome) in linhas[:int(corte_medio * 2)]:
            cor = "magenta" if f_ == n else "orange"
            f.write(f"show sticks, receptor and resi {num}\n")
            f.write(f"color {cor}, receptor and resi {num}\n")
        f.write("zoom ligante, 6\nset cartoon_transparency, 0.6\n")
    print(f"[+] Macro PyMOL: {pml}")
    print("    magenta = unânime nos cinco ligantes | laranja = consenso parcial")


if __name__ == "__main__":
    main()
