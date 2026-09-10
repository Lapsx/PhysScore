#!/usr/bin/env python3
"""Avalia checkpoints no CASF-2016 core, individualmente e como ensemble.

Motivo. O slope de 0,424 é, em boa parte, consequência do MSE e não um defeito do
treino: sob perda quadrática a predição ótima é a média condicional, que tem menos
variância que a verdade. Como o slope de ŷ contra y vale R·σŷ/σy, ele não passa de
**R** por calibração nenhuma — e expandir as predições para forçar slope 1 aumenta
o RMSE. Ou seja, não existe correção pós-hoc: melhorar o slope é melhorar o R.

Ensemble é a forma mais barata de melhorar R sem tocar na arquitetura. A média de
modelos que erram de formas diferentes tem menos variância de erro, e tanto R
quanto o slope sobem juntos por consequência.

Uso:
    python3 ensemble.py pharm_model_weights_best.pth exp_pose_fisica.pth
    python3 ensemble.py ckpt_seed*.pth --saida logs/ensemble.csv
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import PDBbindDataset
from gnn_model import PharmGeometricGNN
from train import load_core_ids, carrega_tipo_ensaio


def montar_teste():
    """O core set do CASF-2016, com o mesmo critério que `train.py` usa."""
    dataset = PDBbindDataset(root=".")
    core = load_core_ids()
    if not core:
        sys.exit("[!] core_set.dat não encontrado — sem conjunto de teste.")
    tipo = carrega_tipo_ensaio()
    idx = {t: i for i, t in enumerate(PharmGeometricGNN.TIPOS_ENSAIO)}

    teste = []
    for g in dataset:
        if getattr(g, "pdb_id", None) in core:
            t = tipo.get(g.pdb_id, "desconhecido")
            g.assay = torch.tensor([[idx[t]]], dtype=torch.long)
            teste.append(g)
    return teste


def carregar(caminho, device):
    """Reconstrói o modelo com as flags que o próprio checkpoint denuncia."""
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    modelo = PharmGeometricGNN(
        node_dim=64, num_gaussians=32, num_layers=4,
        usar_assay=any(k.startswith("assay_embedding") for k in estado),
        usar_pose=any(k.startswith("pose_predictor.") for k in estado),
        usar_direcional=any(".mlp_upd." in k for k in estado),
    ).to(device)
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo


def prever(modelo, loader, device):
    preds, alvos = [], []
    with torch.no_grad():
        for data in loader:
            out, _ = modelo(data.to_homogeneous().to(device),
                            assay=data.assay.to(device))
            preds.append(out.squeeze(-1).cpu())
            alvos.append(data.y.squeeze(-1).cpu())
    return torch.cat(preds).numpy(), torch.cat(alvos).numpy()


def metricas(p, t):
    r = float(np.corrcoef(p, t)[0, 1])
    # slope de PREVISTO contra VERDADEIRO, que é o que o README reporta
    slope = float(np.polyfit(t, p, 1)[0])
    return {
        "R": r,
        "R2": r ** 2,
        "RMSE": float(((p - t) ** 2).mean() ** 0.5),
        "MAE": float(abs(p - t).mean()),
        "slope": slope,
        "sigma_ratio": float(p.std() / t.std()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--saida", default=None, help="CSV com as predições por complexo")
    args = ap.parse_args()

    faltando = [c for c in args.checkpoints if not os.path.exists(c)]
    if faltando:
        sys.exit(f"[!] não encontrado: {', '.join(faltando)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teste = montar_teste()
    loader = DataLoader(teste, batch_size=32, shuffle=False)
    print(f"[*] {len(teste)} complexos do CASF-2016 core · {device}\n")

    todas, alvo = [], None
    print(f"{'checkpoint':<38}{'R':>7}{'RMSE':>8}{'MAE':>7}{'slope':>8}{'sd/sd':>7}")
    print("-" * 75)
    for c in args.checkpoints:
        p, t = prever(carregar(c, device), loader, device)
        alvo = t
        todas.append(p)
        m = metricas(p, t)
        print(f"{os.path.basename(c):<38}{m['R']:>7.3f}{m['RMSE']:>8.3f}"
              f"{m['MAE']:>7.3f}{m['slope']:>8.3f}{m['sigma_ratio']:>7.3f}")

    if len(todas) > 1:
        media = np.mean(todas, axis=0)
        m = metricas(media, alvo)
        print("-" * 75)
        print(f"{'ENSEMBLE (média de %d)' % len(todas):<38}{m['R']:>7.3f}"
              f"{m['RMSE']:>8.3f}{m['MAE']:>7.3f}{m['slope']:>8.3f}"
              f"{m['sigma_ratio']:>7.3f}")

        # O teto do slope, para deixar claro que ele é consequência e não alavanca:
        # com calibração linear perfeita o melhor slope alcançável é o próprio R.
        print(f"\n  slope máximo alcançável por calibração linear = R = {m['R']:.3f}")
        print("  (expandir além disso aumenta o RMSE; ver docstring)")

    print(f"\n  baseline (prever a média): RMSE {float(alvo.std()):.3f}")

    if args.saida:
        import csv
        with open(args.saida, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pdb_id", "alvo"] + [os.path.basename(c) for c in args.checkpoints]
                       + (["ensemble"] if len(todas) > 1 else []))
            ids = [g.pdb_id for g in teste]
            for k, pid in enumerate(ids):
                linha = [pid, alvo[k]] + [p[k] for p in todas]
                if len(todas) > 1:
                    linha.append(float(np.mean([p[k] for p in todas])))
                w.writerow(linha)
        print(f"\n[+] predições por complexo em {args.saida}")


if __name__ == "__main__":
    main()
