#!/usr/bin/env python3
"""Screening power do PhysScore no CASF-2016.

A quarta capacidade que o CASF-2016 mede, e a única que o `README.md` **afirma** sem
nunca ter medido: ele diz que o modelo serve para "ordenar uma biblioteca". Isso não é
o que o Spearman rho de 0,746 mede — aquele é ranking dentro do core set, com poses
cristalográficas e um ligante por alvo. Screening power é a pergunta real da triagem
virtual: dado um alvo, os ligantes que de fato se ligam a ele sobem ao topo de uma
biblioteca de 285 candidatos?

É a mesma estrutura do que a seção 4 encontrou no docking power — uma alegação sobre
utilidade sem medição por trás.

## Protocolo (forward screening power)

57 alvos × 285 ligantes × ~100 poses dockadas = ~1,6 milhão de avaliações.

Para cada alvo: pontuar todas as poses, tomar o **melhor score por ligante**, ordenar
os 285 e perguntar

  * **success top N%** — o melhor binder conhecido do alvo (L1 em `TargetInfo.dat`,
    onde os ligantes vêm ordenados por afinidade) está entre os N% do topo?
  * **EF top N%** — fator de enriquecimento: quantos dos ativos conhecidos aparecem no
    topo, contra o que o acaso daria. EF = 1,0 é o acaso; EF = 10 no top 10% é o
    máximo possível.

Cutoffs: 1% = 3 ligantes, 5% = 14, 10% = 29.

## Bolso

Fixo por alvo, definido pela pose cristalográfica do ligante nativo daquele alvo — os
285 candidatos são dockados no mesmo sítio, então o bolso não pode variar com o
ligante. Mesma correção da seção 4: `build_hetero_graph` recentra o bolso no ligante,
o que aqui daria a cada candidato um bolso sob medida.

Uso:
    python3 screening_power.py --jobs 14
    python3 screening_power.py --limite 5 --jobs 5      # 5 alvos, teste rápido
    python3 screening_power.py --pesos a.pth b.pth      # ensemble
"""

import argparse
import os
import sys
import numpy as np
import torch
from multiprocessing import Pool

from data_processor import PharmGraphBuilder, custom_radius_graph
from gnn_model import PharmGeometricGNN
from docking_power import (_mol_de_bloco_mol2, ler_poses, feats, montar_grafo,
                           DIR_CORESET, PESOS)

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_SCREEN = os.path.join(RAIZ, "CASF-2016", "decoys_screening")
ALVOS_INFO = os.path.join(RAIZ, "CASF-2016", "power_screening", "TargetInfo.dat")


def ler_alvos():
    """{alvo: [ativos]}, com os ativos já ordenados por afinidade decrescente."""
    alvos = {}
    with open(ALVOS_INFO) as fh:
        for linha in fh:
            if linha.startswith('#') or not linha.strip():
                continue
            p = linha.split()
            if len(p) >= 2:
                alvos[p[0]] = p[1:]
    return alvos


def processar(args):
    """Um alvo: pontua os 285 candidatos e devolve o melhor score de cada."""
    alvo, caminhos_pesos, cabeca = args
    torch.set_num_threads(1)
    builder = PharmGraphBuilder(pocket_radius=6.0, interaction_radius=5.0,
                                covalent_radius=2.0)
    try:
        pasta_alvo = os.path.join(DIR_CORESET, alvo)
        prot_pos, prot_feat = builder.parse_protein(
            os.path.join(pasta_alvo, f"{alvo}_protein.pdb"))

        # O bolso vem do ligante nativo DO ALVO, e vale para todos os candidatos.
        nativo = _mol_de_bloco_mol2(
            open(os.path.join(pasta_alvo, f"{alvo}_ligand.mol2")).read())
        if nativo is None:
            return alvo, None, "ligante nativo do alvo ilegivel"
        nat_pos, _ = feats(builder, nativo)

        idx = builder.selecionar_bolso(nat_pos, prot_pos)
        if len(idx) == 0:
            return alvo, None, "bolso vazio"
        pocket_pos = torch.tensor(prot_pos[idx], dtype=torch.float)
        pocket_feat = torch.tensor(prot_feat[idx], dtype=torch.float)
        prot_ei = custom_radius_graph(pocket_pos, r=2.0, loop=False)
        prot_ea = torch.norm(pocket_pos[prot_ei[0]] - pocket_pos[prot_ei[1]],
                             dim=1).view(-1, 1)

        modelos, tem_pose = [], True
        for c in caminhos_pesos:
            est = torch.load(c, map_location="cpu", weights_only=True)
            tp = any(k.startswith("pose_predictor.") for k in est)
            tem_pose = tem_pose and tp
            if cabeca in ("pose", "hibrido") and not tp:
                return alvo, None, f"{c} nao tem cabeca de pose"
            m = PharmGeometricGNN(node_dim=64, num_gaussians=32, num_layers=4,
                                  usar_pose=tp,
                                  usar_direcional=any(".mlp_upd." in k for k in est))
            m.load_state_dict(est)
            modelos.append(m.eval())

        pasta = os.path.join(DIR_SCREEN, alvo)
        melhores = {}
        with torch.no_grad():
            for arq in sorted(os.listdir(pasta)):
                if not arq.endswith(".mol2"):
                    continue
                # nome: <alvo>_<ligante>.mol2
                ligante = arq[:-5].split("_", 1)[1]
                melhor = None       # score reportado para este ligante
                criterio = None     # score usado para ESCOLHER a pose
                for _codigo, mol in ler_poses(os.path.join(pasta, arq)):
                    lig_pos, lig_feat = feats(builder, mol)
                    grafo, _ = montar_grafo(lig_pos, lig_feat, pocket_pos,
                                            pocket_feat, prot_ei, prot_ea, 5.0, 2.0)
                    homo = grafo.to_homogeneous()
                    afins, poses = [], []
                    for m in modelos:
                        if tem_pose:
                            pk, _, pp = m(homo, retornar_pose=True)
                            afins.append(pk)
                            poses.append(pp)
                        else:
                            pk, _ = m(homo)
                            afins.append(pk)
                    v_afin = float(torch.stack(afins).mean().item())
                    v_pose = (float(torch.stack(poses).mean().item())
                              if poses else v_afin)

                    # No modo híbrido a cabeça de POSE escolhe qual pose usar e a
                    # de AFINIDADE dá o valor reportado. É o que um praticante faz:
                    # dockar, escolher a geometria plausível, e só então estimar
                    # potência. Nos outros modos as duas coisas são o mesmo score,
                    # como no protocolo original do CASF — que pressupõe funções de
                    # saída única, tipo Vina.
                    if cabeca == "hibrido":
                        c, v = v_pose, v_afin
                    elif cabeca == "pose":
                        c = v = v_pose
                    else:
                        c = v = v_afin

                    if criterio is None or c > criterio:
                        criterio, melhor = c, v
                if melhor is not None:
                    melhores[ligante] = melhor
        return alvo, melhores, None
    except Exception as exc:
        return alvo, None, f"{type(exc).__name__}: {exc}"


def metricas(resultados, alvos):
    """Success rate e enrichment factor nos cortes de 1%, 5% e 10%."""
    n_lig = max(len(v) for v in resultados.values())
    cortes = {1: max(1, round(n_lig * 0.01)),
              5: max(1, round(n_lig * 0.05)),
              10: max(1, round(n_lig * 0.10))}
    sucesso = {k: [] for k in cortes}
    ef = {k: [] for k in cortes}

    for alvo, scores in resultados.items():
        ativos = [a for a in alvos.get(alvo, []) if a in scores]
        if not ativos:
            continue
        melhor_binder = ativos[0]          # L1: maior afinidade
        ordenado = sorted(scores, key=lambda l: -scores[l])
        for pct, n in cortes.items():
            topo = ordenado[:n]
            sucesso[pct].append(1.0 if melhor_binder in topo else 0.0)
            n_no_topo = sum(1 for a in ativos if a in topo)
            ef[pct].append(n_no_topo / (len(ativos) * pct * 0.01))
    return cortes, sucesso, ef


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--pesos", default=[PESOS], nargs="+")
    ap.add_argument("--cabeca", choices=["afinidade", "pose", "hibrido"],
                    default="afinidade",
                    help="hibrido: a cabeca de pose escolhe a pose, a de afinidade pontua")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    if not os.path.isdir(DIR_SCREEN):
        sys.exit(f"[!] {DIR_SCREEN} não existe — extraia decoys_screening (8,3 GB)")

    alvos = ler_alvos()
    ids = sorted(a for a in alvos if os.path.isdir(os.path.join(DIR_SCREEN, a)))
    if args.limite:
        ids = ids[:args.limite]

    print(f"[*] {len(ids)} alvos · {args.jobs} processos")
    print(f"[*] {len(args.pesos)} checkpoint(s), cabeça de {args.cabeca}")

    resultados, falhas = {}, []
    with Pool(args.jobs) as pool:
        for i, (alvo, scores, erro) in enumerate(
                pool.imap_unordered(processar,
                                    [(a, args.pesos, args.cabeca) for a in ids]), 1):
            if erro:
                falhas.append((alvo, erro))
            else:
                resultados[alvo] = scores
            print(f"    {i}/{len(ids)}  ({len(falhas)} falhas)", flush=True)

    if falhas:
        print(f"\n[!] {len(falhas)} falhas:")
        for a, e in falhas[:5]:
            print(f"    {a}: {e}")
    if not resultados:
        sys.exit("[!] nenhum alvo processado")

    cortes, sucesso, ef = metricas(resultados, alvos)
    print(f"\nScreening power — CASF-2016, {len(resultados)} alvos, "
          f"cabeça '{args.cabeca}'")
    print(f"cortes: top1% = {cortes[1]} ligantes · top5% = {cortes[5]} · "
          f"top10% = {cortes[10]}\n")
    print(f"{'corte':<10}{'success %':>12}{'EF':>10}{'EF do acaso':>14}")
    print("-" * 46)
    for pct in (1, 5, 10):
        print(f"top {pct}%{'':<4}{100*np.mean(sucesso[pct]):>12.1f}"
              f"{np.mean(ef[pct]):>10.2f}{1.0:>14.2f}")
    print("-" * 46)
    print(f"EF máximo possível no top {1}%: {100/1:.0f} · "
          f"top 5%: {100/5:.0f} · top 10%: {100/10:.0f}")

    if args.saida:
        import csv
        with open(args.saida, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["alvo", "ligante", "score", "eh_ativo"])
            for alvo, scores in sorted(resultados.items()):
                ativos = set(alvos.get(alvo, []))
                for lig, v in sorted(scores.items(), key=lambda kv: -kv[1]):
                    w.writerow([alvo, lig, v, int(lig in ativos)])
        print(f"\n[+] scores por par alvo–ligante em {args.saida}")


if __name__ == "__main__":
    main()
