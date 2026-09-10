#!/usr/bin/env python3
"""Docking power das 34 funções de referência do CASF-2016, sob condições idênticas.

O pacote traz, em `power_docking/examples/`, o score de cada função clássica para
as MESMAS poses que `docking_power.py` avalia. Isso permite para docking power a
mesma comparação que o `README.md` faz para scoring power: recomputar todo mundo
com um único código de métrica, em vez de citar números publicados sob protocolos
possivelmente diferentes.

A convenção de sinal é determinada por função, empiricamente: cada uma é avaliada
nas duas orientações e fica com a melhor. Funções de energia (Vina, PLP) reportam
"mais negativo é melhor" e funções de afinidade o contrário; escolher errado daria
um docking power abaixo do acaso e produziria um ranking sem sentido.
"""

import csv
import os
import sys
import numpy as np
from scipy.stats import spearmanr

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_EX = os.path.join(RAIZ, "CASF-2016", "power_docking", "examples")
DIR_DECOYS = os.path.join(RAIZ, "CASF-2016", "decoys_docking")
CUT_RMSD = 2.0


def ler_rmsd(pdb_id):
    tabela = {}
    with open(os.path.join(DIR_DECOYS, f"{pdb_id}_rmsd.dat")) as fh:
        for linha in fh:
            p = linha.split()
            if len(p) >= 2 and not linha.startswith('#'):
                try:
                    tabela[p[0]] = float(p[1])
                except ValueError:
                    pass
    return tabela


def metricas(por_complexo):
    """Idêntica à de docking_power.py: ordena por score decrescente e pergunta se
    alguma das N primeiras é near-native."""
    topo = {1: [], 2: [], 3: []}
    rho = []
    for linhas in por_complexo.values():
        ordenado = sorted(linhas, key=lambda d: -d["score"])
        for n in (1, 2, 3):
            topo[n].append(1.0 if min(d["rmsd"] for d in ordenado[:n]) <= CUT_RMSD else 0.0)
        s = [d["score"] for d in linhas]
        r = [d["rmsd"] for d in linhas]
        if len(set(s)) > 1:
            c = spearmanr(s, r).correlation
            if not np.isnan(c):
                rho.append(-c)
    return {
        "top1": 100 * np.mean(topo[1]),
        "top2": 100 * np.mean(topo[2]),
        "top3": 100 * np.mean(topo[3]),
        "rho": float(np.mean(rho)) if rho else float("nan"),
        "n": len(por_complexo),
    }


def carregar_referencia(nome, ids, restringir=None):
    """`restringir`: {pdb_id: {codigos}} para avaliar todas as funções exatamente
    nas mesmas poses. 2,5% dos decoys falham no parse de mol2 do nosso lado, e sem
    isso as referências seriam medidas num conjunto ligeiramente maior — uma
    diferença pequena, mas gratuita de eliminar."""
    pasta = os.path.join(DIR_EX, nome)
    por_complexo = {}
    for pdb_id in ids:
        caminho = os.path.join(pasta, f"{pdb_id}_score.dat")
        if not os.path.exists(caminho):
            continue
        if restringir is not None and pdb_id not in restringir:
            continue
        rmsds = ler_rmsd(pdb_id)
        if restringir is not None:
            rmsds = {k: v for k, v in rmsds.items() if k in restringir[pdb_id]}
        linhas = []
        with open(caminho) as fh:
            for linha in fh:
                p = linha.split()
                if len(p) < 2 or linha.startswith('#'):
                    continue
                try:
                    valor = float(p[1])
                except ValueError:
                    continue
                if p[0] in rmsds:
                    linhas.append({"score": valor, "rmsd": rmsds[p[0]]})
        if len(linhas) >= 5:
            por_complexo[pdb_id] = linhas
    return por_complexo


def melhor_orientacao(por_complexo):
    """Avalia nas duas convenções de sinal e devolve a melhor, com o sinal usado."""
    direto = metricas(por_complexo)
    invertido_dados = {k: [{"score": -d["score"], "rmsd": d["rmsd"]} for d in v]
                       for k, v in por_complexo.items()}
    invertido = metricas(invertido_dados)
    if invertido["top1"] > direto["top1"]:
        return invertido, "-"
    return direto, "+"


def carregar_pharmxai(caminho_csv):
    por_complexo, codigos = {}, {}
    with open(caminho_csv) as fh:
        for linha in csv.DictReader(fh):
            por_complexo.setdefault(linha["pdb_id"], []).append(
                {"score": float(linha["modelo"]), "rmsd": float(linha["rmsd"])})
            codigos.setdefault(linha["pdb_id"], set()).add(linha["codigo"])
    return por_complexo, codigos


def main():
    csv_pharm = sys.argv[1] if len(sys.argv) > 1 else "logs/dock_nativa.csv"
    if not os.path.isdir(DIR_EX):
        sys.exit(f"[!] {DIR_EX} não existe")

    ids = sorted(f[:-len("_rmsd.dat")] for f in os.listdir(DIR_DECOYS)
                 if f.endswith("_rmsd.dat"))

    restringir = None
    dados_pharm = None
    if os.path.exists(csv_pharm):
        dados_pharm, restringir = carregar_pharmxai(csv_pharm)

    resultados = []
    for nome in sorted(os.listdir(DIR_EX)):
        if not os.path.isdir(os.path.join(DIR_EX, nome)):
            continue
        dados = carregar_referencia(nome, ids, restringir)
        if not dados:
            continue
        m, sinal = melhor_orientacao(dados)
        resultados.append((nome, m, sinal))

    if dados_pharm is not None:
        resultados.append(("PharmXAI-3D (este)", metricas(dados_pharm), "+"))

    resultados.sort(key=lambda t: -t[1]["top1"])

    print(f"\nDocking power — CASF-2016, near-native = RMSD <= {CUT_RMSD} Å")
    print(f"Todas recomputadas com a mesma métrica. PharmXAI de: {csv_pharm}\n")
    print(f"{'#':>3}  {'funcao':<24}{'top1 %':>8}{'top2 %':>8}{'top3 %':>8}{'rho':>8}{'n':>6}{'sinal':>7}")
    print("-" * 74)
    for i, (nome, m, sinal) in enumerate(resultados, 1):
        marca = "  <--" if "PharmXAI" in nome else ""
        print(f"{i:>3}  {nome:<24}{m['top1']:>8.1f}{m['top2']:>8.1f}{m['top3']:>8.1f}"
              f"{m['rho']:>8.3f}{m['n']:>6}{sinal:>7}{marca}")
    print("-" * 74)
    ref = [m["top1"] for nome, m, _ in resultados if "PharmXAI" not in nome]
    print(f"     {'mediana das referencias':<24}{np.median(ref):>8.1f}")


if __name__ == "__main__":
    main()
