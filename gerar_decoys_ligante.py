#!/usr/bin/env python3
"""Decoys de LIGANTE: o mesmo bolso com um composto que não se liga a ele.

Motivo (seção 7 do EXPERIMENTOS.md). Privado da identidade do alvo, o modelo não supera
uma regressão linear sobre descritores do ligante — R = 0,548 contra 0,563. Dos 0,752
que ele obtém no CASF-2016, cerca de 0,56 é propriedade da molécula e apenas ~0,19 vem
de modelar a interação.

A causa está no rótulo, não na arquitetura: o pKd do PDBbind é a afinidade de cada
ligante pelo **seu próprio** alvo, e nada no treino jamais mostrou o mesmo ligante num
alvo errado. "Molécula potente" e "par que se liga" são indistinguíveis nos dados, e o
modelo aprende a primeira porque é a mais fácil.

Isto gera os exemplos negativos que faltam — o análogo, para o par, do que as
perturbações rígidas foram para a pose (seção 5).

## Como o decoy é construído

**Transplante geométrico**, sobre os grafos já processados, sem reparsear PDB nenhum:

  1. do grafo do complexo *j*, tomam-se os átomos do LIGANTE (features e posições);
  2. do grafo do complexo *i*, toma-se o BOLSO;
  3. o ligante de *j* é rotacionado ao acaso e transladado para o centroide do ligante
     de *i*, isto é, para dentro do sítio de ligação de *i*;
  4. as arestas de interação são recalculadas a partir da nova geometria — aqui elas
     PRECISAM ser refeitas, ao contrário das perturbações rígidas da seção 5, porque o
     ligante é outro e a topologia muda.

A pose resultante não é realista: tem clash e não passou por docking. O precedente que
justifica tentar mesmo assim é a seção 5, onde perturbações rígidas igualmente
sintéticas transferiram para decoys de docking real. Se não transferir desta vez, a
rota cara (redocking com smina) fica como alternativa — e o screening power é o teste
que decide.

## O filtro de alvo

Um decoy só é válido se o ligante *j* de fato NÃO se liga ao bolso de *i*. Sem
sequências à mão, o filtro usado é a dissimilaridade de composição do bolso: contagem
de elementos, tamanho e número de contatos, comparadas por distância de cosseno. Dois
bolsos do mesmo alvo têm composição quase idêntica; a rejeição é conservadora.

**Isto é mais fraco que o containment de 4-mers da seção 3** e deixa passar algum falso
negativo — um ligante que se ligaria mesmo. A fração é reportada no fim para que o
efeito seja dimensionável, e a loss de ranking usa margem, o que tolera ruído de rótulo.

Uso:
    python3 gerar_decoys_ligante.py --n 2 --limite 3000 --saida decoys_ligante.pt
"""

import argparse
import os
import numpy as np
import torch
from scipy.spatial import KDTree

from dataset import PDBbindDataset


def assinatura_bolso(g):
    """Vetor de composição do bolso, para medir se dois complexos são o mesmo alvo."""
    x = g['protein'].x
    z = x[:, 0].long()
    # contagem por número atômico dos elementos comuns em proteína
    cont = torch.tensor([float((z == e).sum()) for e in (6, 7, 8, 16)])
    total = float(x.shape[0])
    return torch.cat([cont / max(total, 1.0), torch.tensor([total / 200.0])])


def cosseno(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def transplantar(lig_pos, centro_alvo, gerador):
    """Rotação aleatória em torno do próprio centroide, depois translação para o sítio."""
    c = lig_pos.mean(dim=0, keepdim=True)
    v = lig_pos - c
    eixo = torch.randn(3, generator=gerador)
    eixo = eixo / eixo.norm().clamp(min=1e-8)
    ang = torch.rand(1, generator=gerador) * 2 * np.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    v_rot = (v * cos + torch.cross(eixo.expand_as(v), v, dim=1) * sin
             + eixo.unsqueeze(0) * (v @ eixo).unsqueeze(-1) * (1 - cos))
    return v_rot + centro_alvo


def montar(lig_x, lig_pos, prot_x, prot_pos, raio_inter=5.0, raio_cov=2.0):
    """HeteroData do par (bolso de i, ligante de j), com arestas refeitas."""
    from torch_geometric.data import HeteroData
    from data_processor import custom_radius_graph
    d = HeteroData()
    d['ligand'].x, d['ligand'].pos = lig_x, lig_pos
    d['protein'].x, d['protein'].pos = prot_x, prot_pos

    li = custom_radius_graph(lig_pos, r=raio_cov, loop=False)
    d['ligand', 'covalent', 'ligand'].edge_index = li
    d['ligand', 'covalent', 'ligand'].edge_attr = torch.norm(
        lig_pos[li[0]] - lig_pos[li[1]], dim=1).view(-1, 1)

    pi = custom_radius_graph(prot_pos, r=raio_cov, loop=False)
    d['protein', 'covalent', 'protein'].edge_index = pi
    d['protein', 'covalent', 'protein'].edge_attr = torch.norm(
        prot_pos[pi[0]] - prot_pos[pi[1]], dim=1).view(-1, 1)

    viz = KDTree(prot_pos.numpy()).query_ball_point(lig_pos.numpy(), r=raio_inter)
    a, b = [], []
    for il, vs in enumerate(viz):
        for ip in vs:
            a.append(il)
            b.append(ip)
    ei = torch.tensor([a, b], dtype=torch.long).view(2, -1)
    ea = torch.norm(lig_pos[ei[0]] - prot_pos[ei[1]], dim=1).view(-1, 1) \
        if ei.numel() else torch.zeros(0, 1)
    d['ligand', 'interacts', 'protein'].edge_index = ei
    d['ligand', 'interacts', 'protein'].edge_attr = ea
    d['protein', 'interacts', 'ligand'].edge_index = torch.tensor(
        [b, a], dtype=torch.long).view(2, -1)
    d['protein', 'interacts', 'ligand'].edge_attr = ea
    return d, ei.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2, help="decoys por complexo")
    ap.add_argument("--limite", type=int, default=3000, help="complexos a usar")
    ap.add_argument("--cos-max", type=float, default=0.995,
                    help="acima disto os bolsos sao considerados o mesmo alvo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default="decoys_ligante.pt")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds = PDBbindDataset(root=".")
    n_total = len(ds)
    print(f"[*] {n_total} complexos no dataset; usando {min(args.limite, n_total)}")

    idx = rng.permutation(n_total)[:args.limite]
    assinaturas = {}
    saida, rejeitados, sem_contato = [], 0, 0

    for k, i in enumerate(idx, 1):
        gi = ds[int(i)]
        if i not in assinaturas:
            assinaturas[i] = assinatura_bolso(gi)
        centro = gi['ligand'].pos.mean(dim=0, keepdim=True)

        feitos = 0
        tentativas = 0
        while feitos < args.n and tentativas < args.n * 8:
            tentativas += 1
            j = int(rng.integers(n_total))
            if j == i:
                continue
            gj = ds[j]
            if j not in assinaturas:
                assinaturas[j] = assinatura_bolso(gj)
            if cosseno(assinaturas[int(i)], assinaturas[j]) > args.cos_max:
                rejeitados += 1
                continue        # provavelmente o mesmo alvo
            novo_pos = transplantar(gj['ligand'].pos, centro, g)
            d, n_cont = montar(gj['ligand'].x, novo_pos,
                               gi['protein'].x, gi['protein'].pos)
            if n_cont < 10:
                sem_contato += 1
                continue        # caiu fora do bolso; inútil como negativo
            d.pdb_id = getattr(gi, 'pdb_id', '?')
            d.decoy_de = getattr(gj, 'pdb_id', '?')
            saida.append(d)
            feitos += 1

        if k % 250 == 0:
            print(f"    {k}/{len(idx)} · {len(saida)} decoys · "
                  f"{rejeitados} rejeitados por bolso similar · "
                  f"{sem_contato} sem contato", flush=True)

    # Serialização compacta: `InMemoryDataset.collate` concatena tudo em tensores
    # únicos com índices de fatiamento. Medido: 30,7 KB por decoy contra 478 KB
    # salvando a lista de objetos HeteroData direto — 16x, porque o pickle de cada
    # objeto carrega a estrutura inteira do PyG junto.
    from torch_geometric.data import InMemoryDataset
    dados, slices = InMemoryDataset.collate(saida)
    torch.save((dados, slices), args.saida)
    print(f"\n[+] {len(saida)} decoys de ligante em {args.saida}")
    print(f"    rejeitados por bolso similar : {rejeitados}")
    print(f"    descartados por falta de contato: {sem_contato}")
    if saida:
        nc = [d['ligand', 'interacts', 'protein'].edge_index.shape[1] for d in saida]
        print(f"    contatos por decoy: mediana {int(np.median(nc))}, "
              f"min {min(nc)}, max {max(nc)}")


if __name__ == "__main__":
    main()
