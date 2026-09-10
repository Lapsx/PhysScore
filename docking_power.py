#!/usr/bin/env python3
"""Docking power do PharmXAI-3D no CASF-2016 — diagnóstico do modelo atual.

O `README.md` reporta scoring power (R = 0,754 em poses cristalográficas). Esta é
a pergunta independente: dado um sítio e ~92 poses candidatas do MESMO ligante,
o modelo coloca a pose correta no topo? É o que separa uma função de re-scoring
de uma scoring function utilizável em docking.

Nada aqui treina nem escreve pesos. Só lê `pharm_model_weights_best.pth`.

## O bolso precisa ser fixo

`PharmGraphBuilder.build_hetero_graph` define o bolso como os átomos de proteína a
`pocket_radius` do LIGANTE, recalculado a cada chamada. Para afinidade em pose
cristalográfica isso é correto. Para docking power é fatal: mover o ligante move o
bolso junto, e uma pose deslocada ganha um bolso perfeitamente formado ao redor de
si — a informação "esta pose está no lugar errado" é apagada na construção do grafo,
antes de a rede ver qualquer coisa.

Aqui o conjunto de átomos do bolso é escolhido UMA vez por complexo e reusado por
todas as poses. Só as arestas de interação são recalculadas, que é o que de fato
depende da pose.

Dois modos, porque a escolha do bolso é ela própria uma hipótese:

  `nativa`  bolso a 6 Å da pose cristalográfica. Reproduz a distribuição em que o
            modelo foi treinado, mas penaliza poses distantes por truncamento: elas
            perdem contatos porque o bolso acaba, não só porque a pose é ruim.

  `uniao`   bolso a 6 Å da união de todas as poses do complexo. Idêntico para todas
            e sem truncamento, ao custo de um bolso maior que o do treino.

Rodar os dois é o controle: se o docking power desaparece em `uniao`, o que o modo
`nativa` mediu era artefato de truncamento, não geometria aprendida.

## Baselines triviais

Reporta, junto do modelo, duas funções que não aprenderam nada:

  `n_contatos`  número de arestas de interação
  `lj`          o mesmo proxy de Lennard-Jones que `train.py` usa como alvo auxiliar

O readout entrega `log1p(n_contatos)` explicitamente ao MLP (`gnn_model.py`), então
o modelo tem acesso direto à contagem. Se ele não superar `n_contatos`, o docking
power que exibir é contagem de contatos com passos extras — o mesmo diagnóstico que
a regressão de 8 features de contagem deu para o readout antigo.

Uso:
    python3 docking_power.py                      # bolso nativa
    python3 docking_power.py --bolso uniao
    python3 docking_power.py --limite 20          # 20 complexos, para teste rápido
"""

import argparse
import os
import sys
import numpy as np
import torch
from multiprocessing import Pool
from scipy.spatial import KDTree
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from torch_geometric.data import HeteroData

from data_processor import PharmGraphBuilder, custom_radius_graph
from gnn_model import PharmGeometricGNN

RDLogger.DisableLog('rdApp.*')

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_CASF = os.path.join(RAIZ, "CASF-2016")
DIR_DECOYS = os.path.join(DIR_CASF, "decoys_docking")
DIR_CORESET = os.path.join(DIR_CASF, "coreset")
PESOS = os.path.join(RAIZ, "pharm_model_weights_best.pth")

CUT_RMSD = 2.0      # define pose near-native, como no protocolo oficial
SIGMA_LJ = 3.5      # o mesmo de calculate_lj_potential em train.py


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------

def _mol_de_bloco_mol2(bloco):
    """Aplica a mesma cascata tolerante de `_ler_ligante`, mas sobre um bloco."""
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    mol = Chem.MolFromMol2Block(bloco, removeHs=False, sanitize=False)
    if mol is None:
        return None
    if Chem.SanitizeMol(mol, sanitizeOps=ops, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    PharmGraphBuilder._marcar_doadores(mol)
    PharmGraphBuilder._marcar_cargas(mol, eh_proteina=False)
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    return mol


def ler_poses(caminho_mol2):
    """[(codigo, mol)] de um .mol2 multi-molécula. O código casa com o _rmsd.dat."""
    with open(caminho_mol2) as fh:
        texto = fh.read()
    saida = []
    for pedaco in texto.split('@<TRIPOS>MOLECULE')[1:]:
        bloco = '@<TRIPOS>MOLECULE' + pedaco
        linhas = bloco.splitlines()
        if len(linhas) < 2:
            continue
        codigo = linhas[1].strip()
        mol = _mol_de_bloco_mol2(bloco)
        if mol is not None and mol.GetNumConformers() > 0 and codigo:
            saida.append((codigo, mol))
    return saida


def ler_rmsd(caminho):
    """{codigo: rmsd} do arquivo _rmsd.dat."""
    tabela = {}
    with open(caminho) as fh:
        for linha in fh:
            partes = linha.split()
            if len(partes) < 2 or linha.startswith('#'):
                continue
            try:
                tabela[partes[0]] = float(partes[1])
            except ValueError:
                continue
    return tabela


def feats(builder, mol):
    pos = np.asarray(mol.GetConformer().GetPositions())
    f = np.asarray([builder.get_atom_features(a) for a in mol.GetAtoms()], dtype=float)
    return pos, f


# ----------------------------------------------------------------------------
# Grafo com bolso fixo
# ----------------------------------------------------------------------------

def montar_grafo(lig_pos, lig_feat, pocket_pos, pocket_feat,
                 prot_edge_idx, prot_edge_attr, raio_inter, raio_cov):
    """HeteroData de uma pose sobre um bolso JÁ escolhido.

    As arestas covalentes da proteína dependem só do bolso, que é fixo, então
    chegam prontas — é o cálculo mais caro e seria repetido ~92 vezes à toa.

    A ordem de criação das arestas define os índices de `edge_type` após
    `to_homogeneous()`: 0 e 1 covalentes, 2 e 3 de interação, como `train.py`
    documenta em TIPOS_NAO_LIGADOS.
    """
    data = HeteroData()
    lig_pos_t = torch.tensor(lig_pos, dtype=torch.float)
    data['ligand'].x = torch.tensor(lig_feat, dtype=torch.float)
    data['ligand'].pos = lig_pos_t
    data['protein'].x = pocket_feat
    data['protein'].pos = pocket_pos

    lig_edge_idx = custom_radius_graph(lig_pos_t, r=raio_cov, loop=False)
    lig_edge_attr = torch.norm(
        lig_pos_t[lig_edge_idx[0]] - lig_pos_t[lig_edge_idx[1]], dim=1).view(-1, 1)
    data['ligand', 'covalent', 'ligand'].edge_index = lig_edge_idx
    data['ligand', 'covalent', 'ligand'].edge_attr = lig_edge_attr

    data['protein', 'covalent', 'protein'].edge_index = prot_edge_idx
    data['protein', 'covalent', 'protein'].edge_attr = prot_edge_attr

    pocket_np = pocket_pos.numpy()
    vizinhos = KDTree(pocket_np).query_ball_point(lig_pos, r=raio_inter)
    li, pi, dist = [], [], []
    for idx_l, viz in enumerate(vizinhos):
        for idx_p in viz:
            li.append(idx_l)
            pi.append(idx_p)
            dist.append(np.linalg.norm(lig_pos[idx_l] - pocket_np[idx_p]))

    inter_idx = torch.tensor([li, pi], dtype=torch.long).view(2, -1)
    inter_attr = torch.tensor(dist, dtype=torch.float).view(-1, 1)
    data['ligand', 'interacts', 'protein'].edge_index = inter_idx
    data['ligand', 'interacts', 'protein'].edge_attr = inter_attr
    data['protein', 'interacts', 'ligand'].edge_index = torch.tensor(
        [pi, li], dtype=torch.long).view(2, -1)
    data['protein', 'interacts', 'ligand'].edge_attr = inter_attr

    return data, np.asarray(dist)


def lj_medio(distancias, sigma=SIGMA_LJ):
    """Mesma forma de calculate_lj_potential: média por aresta, log-comprimida.

    Devolvido NEGADO, para que 'maior é melhor' valha para todos os scores e o
    ordenamento seja o mesmo em todas as colunas.
    """
    if distancias.size == 0:
        return 0.0
    r = np.clip(distancias, 1.5, None)
    termo = (sigma / r) ** 6
    media = float((termo ** 2 - 2 * termo).mean())
    return -float(np.sign(media) * np.log1p(abs(media)))


# ----------------------------------------------------------------------------
# Um complexo
# ----------------------------------------------------------------------------

def processar(args):
    pdb_id, modo_bolso, caminhos_pesos, cabeca = args
    torch.set_num_threads(1)
    builder = PharmGraphBuilder(pocket_radius=6.0, interaction_radius=5.0,
                                covalent_radius=2.0)
    try:
        pasta = os.path.join(DIR_CORESET, pdb_id)
        prot_pos, prot_feat = builder.parse_protein(
            os.path.join(pasta, f"{pdb_id}_protein.pdb"))

        # A pose cristalográfica é lida do .mol2 do coreset pelo MESMO parser dos
        # decoys, e não do .sdf. Não é preciosismo: pelo .sdf ela chega com 49
        # átomos onde o decoy do mesmo ligante tem 26 (o RemoveHs não pega os
        # hidrogênios do .sdf sob sanitização tolerante). Com quase o dobro de
        # átomos ela ganha quase o dobro de contatos, e passa a ser identificável
        # por contagem em vez de por geometria — o baseline `n_contatos` subia a
        # 96,1% de top1, contra 24,9% sem ela. Um vazamento, não um resultado.
        bloco_nativo = open(os.path.join(pasta, f"{pdb_id}_ligand.mol2")).read()
        mol_nativo = _mol_de_bloco_mol2(bloco_nativo)
        if mol_nativo is None:
            return pdb_id, None, "pose cristalografica ilegivel"
        nat_pos, nat_feat = feats(builder, mol_nativo)

        poses = ler_poses(os.path.join(DIR_DECOYS, f"{pdb_id}_decoys.mol2"))
        rmsds = ler_rmsd(os.path.join(DIR_DECOYS, f"{pdb_id}_rmsd.dat"))
        poses = [(c, m) for c, m in poses if c in rmsds]
        if len(poses) < 5:
            return pdb_id, None, "poses insuficientes"

        dados_poses = [(c, *feats(builder, m)) for c, m in poses]

        # A pose cristalográfica é candidata, e precisa ser incluída à mão.
        #
        # Ela aparece no _rmsd.dat como `{pdb}_ligand` com RMSD 0,0, e as 34
        # funções de referência do pacote a pontuam — mas ela não está no
        # _decoys.mol2, e sim em coreset/. Deixá-la de fora tirava do conjunto
        # justamente a pose mais fácil de acertar, e só do nosso lado: as
        # referências mantinham um acerto garantido que nós não tínhamos.
        # Medido: as 285 poses ausentes por esse motivo eram 100% near-native,
        # contra 24,1% da população de decoys.
        codigo_nativo = f"{pdb_id}_ligand"
        if codigo_nativo in rmsds:
            dados_poses.append((codigo_nativo, nat_pos, nat_feat))

        # --- a escolha do bolso, feita UMA vez ---
        if modo_bolso == "uniao":
            referencia = np.vstack([nat_pos] + [p for _, p, _ in dados_poses])
        else:
            referencia = nat_pos
        idx_bolso = builder.selecionar_bolso(referencia, prot_pos)
        if len(idx_bolso) == 0:
            return pdb_id, None, "bolso vazio"

        pocket_pos = torch.tensor(prot_pos[idx_bolso], dtype=torch.float)
        pocket_feat = torch.tensor(prot_feat[idx_bolso], dtype=torch.float)
        prot_edge_idx = custom_radius_graph(pocket_pos, r=2.0, loop=False)
        prot_edge_attr = torch.norm(
            pocket_pos[prot_edge_idx[0]] - pocket_pos[prot_edge_idx[1]], dim=1).view(-1, 1)

        # Cada checkpoint diz que arquitetura é: `pose_predictor.` só existe com a
        # cabeça de pose, `.mlp_upd.` só em CamadaDirecional. Um ensemble pode
        # misturar arquiteturas — é justamente daí que vem o ganho dele.
        modelos = []
        for caminho in caminhos_pesos:
            estado = torch.load(caminho, map_location="cpu", weights_only=True)
            tem_pose = any(k.startswith("pose_predictor.") for k in estado)
            if cabeca == "pose" and not tem_pose:
                return pdb_id, None, f"{caminho} nao tem cabeca de pose"
            m = PharmGeometricGNN(
                node_dim=64, num_gaussians=32, num_layers=4, usar_pose=tem_pose,
                usar_direcional=any(".mlp_upd." in k for k in estado))
            m.load_state_dict(estado)
            modelos.append(m.eval())
        tem_pose = all(any(k.startswith("pose_predictor.")
                           for k in torch.load(c, map_location="cpu", weights_only=True))
                       for c in caminhos_pesos)

        linhas = []
        with torch.no_grad():
            for codigo, lig_pos, lig_feat in dados_poses:
                grafo, dist = montar_grafo(
                    lig_pos, lig_feat, pocket_pos, pocket_feat,
                    prot_edge_idx, prot_edge_attr, 5.0, 2.0)
                homo = grafo.to_homogeneous()
                # A média é sobre o SCORE, não sobre os pesos: os modelos têm
                # arquiteturas diferentes e seus pesos não são comensuráveis.
                saidas = []
                for m in modelos:
                    if tem_pose:
                        p_k, _, p_p = m(homo, retornar_pose=True)
                        saidas.append(p_p if cabeca == "pose" else p_k)
                    else:
                        p_k, _ = m(homo)
                        saidas.append(p_k)
                escolhido = torch.stack(saidas).mean(dim=0)
                pkd = pose = escolhido
                # A cabeça de pose é a que foi treinada para ordenar poses; a de
                # afinidade responde outra pergunta. Comparar as duas no mesmo
                # checkpoint diz se a física acrescentou algo ou se só reembalou
                # o que o pKd já fazia.
                linhas.append({
                    "codigo": codigo,
                    "rmsd": rmsds[codigo],
                    "modelo": float(escolhido.item()),
                    "n_contatos": float(dist.size),
                    "lj": lj_medio(dist),
                })
        return pdb_id, linhas, None
    except Exception as exc:
        return pdb_id, None, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------

def metricas(por_complexo, coluna):
    """Protocolo do CASF-2016: ordena por score decrescente e pergunta se alguma
    das N primeiras é near-native (RMSD <= 2 Å)."""
    topo = {1: [], 2: [], 3: []}
    rho, rmsd_top1 = [], []
    for linhas in por_complexo.values():
        ordenado = sorted(linhas, key=lambda d: -d[coluna])
        for n in (1, 2, 3):
            topo[n].append(1.0 if min(d["rmsd"] for d in ordenado[:n]) <= CUT_RMSD else 0.0)
        rmsd_top1.append(ordenado[0]["rmsd"])
        s = [d[coluna] for d in linhas]
        r = [d["rmsd"] for d in linhas]
        if len(set(s)) > 1:
            c = spearmanr(s, r).correlation
            if not np.isnan(c):
                rho.append(-c)   # score alto deve casar com rmsd baixo
    return {
        "top1": 100 * np.mean(topo[1]),
        "top2": 100 * np.mean(topo[2]),
        "top3": 100 * np.mean(topo[3]),
        "rho": float(np.mean(rho)) if rho else float("nan"),
        "rmsd_medio_top1": float(np.mean(rmsd_top1)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bolso", choices=["nativa", "uniao"], default="nativa")
    ap.add_argument("--limite", type=int, default=None,
                    help="usa só os N primeiros complexos (teste rápido)")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--saida", default=None, help="CSV com o score de cada pose")
    ap.add_argument("--pesos", default=[PESOS], nargs="+",
                    help="um ou mais checkpoints; vários = ensemble (média do score)")
    ap.add_argument("--cabeca", choices=["afinidade", "pose"], default="afinidade",
                    help="qual saída usa como score de pose")
    args = ap.parse_args()

    if not os.path.isdir(DIR_DECOYS):
        sys.exit(f"[!] {DIR_DECOYS} não existe — extraia decoys_docking do CASF-2016.tar.gz")

    SUFIXO = "_rmsd.dat"
    ids = sorted(f[:-len(SUFIXO)] for f in os.listdir(DIR_DECOYS)
                 if f.endswith(SUFIXO))
    if args.limite:
        ids = ids[:args.limite]

    print(f"[*] {len(ids)} complexos · bolso '{args.bolso}' · {args.jobs} processos")
    print(f"[*] {len(args.pesos)} checkpoint(s): {', '.join(os.path.basename(p) for p in args.pesos)}")
    print(f"[*] score pela cabeça de {args.cabeca}")

    por_complexo, falhas = {}, []
    with Pool(args.jobs) as pool:
        for i, (pdb_id, linhas, erro) in enumerate(
                pool.imap_unordered(
                    processar,
                    [(p, args.bolso, args.pesos, args.cabeca) for p in ids]), 1):
            if erro:
                falhas.append((pdb_id, erro))
            else:
                por_complexo[pdb_id] = linhas
            if i % 25 == 0 or i == len(ids):
                print(f"    {i}/{len(ids)}  ({len(falhas)} falhas)", flush=True)

    if falhas:
        print(f"\n[!] {len(falhas)} falhas:")
        for p, e in falhas[:5]:
            print(f"    {p}: {e}")

    if not por_complexo:
        sys.exit("[!] nenhum complexo processado")

    n_poses = sum(len(v) for v in por_complexo.values())
    print(f"\n[+] {len(por_complexo)} complexos, {n_poses} poses "
          f"({n_poses / len(por_complexo):.0f} por complexo)")

    print(f"\nDocking power — CASF-2016, near-native = RMSD <= {CUT_RMSD} Å, "
          f"bolso '{args.bolso}', cabeça '{args.cabeca}'\n")
    print(f"{'score':<14}{'top1 %':>9}{'top2 %':>9}{'top3 %':>9}{'rho medio':>12}{'RMSD top1':>12}")
    print("-" * 65)
    for coluna, rotulo in (("modelo", f"modelo ({args.cabeca})"),
                           ("n_contatos", "n_contatos"),
                           ("lj", "LJ (proxy)")):
        m = metricas(por_complexo, coluna)
        print(f"{rotulo:<14}{m['top1']:>9.1f}{m['top2']:>9.1f}{m['top3']:>9.1f}"
              f"{m['rho']:>12.3f}{m['rmsd_medio_top1']:>12.2f}")

    aleatorio = 100 * np.mean([
        np.mean([1.0 if d["rmsd"] <= CUT_RMSD else 0.0 for d in linhas])
        for linhas in por_complexo.values()])
    print("-" * 65)
    print(f"{'acaso (top1)':<14}{aleatorio:>9.1f}"
          "     — fração de poses near-native no conjunto")

    if args.saida:
        import csv
        with open(args.saida, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pdb_id", "codigo", "rmsd", "modelo", "n_contatos", "lj"])
            for pdb_id, linhas in sorted(por_complexo.items()):
                for d in linhas:
                    w.writerow([pdb_id, d["codigo"], d["rmsd"],
                                d["modelo"], d["n_contatos"], d["lj"]])
        print(f"\n[+] scores por pose em {args.saida}")


if __name__ == "__main__":
    main()
