#!/usr/bin/env python3
"""Screening power das 34 funções de referência, sob condições idênticas.

Mesmo papel de `comparar_docking.py`: o pacote CASF-2016 traz, em
`power_screening/examples/`, o score de cada função clássica para os mesmos pares
alvo–ligante, e recomputar todas com um único código de métrica evita comparar
números publicados sob protocolos possivelmente diferentes.

Convenção de sinal determinada empiricamente por função, como no docking.
"""
import csv
import os
import sys
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_EX = os.path.join(RAIZ, "CASF-2016", "power_screening", "examples")
ALVOS_INFO = os.path.join(RAIZ, "CASF-2016", "power_screening", "TargetInfo.dat")


def ler_alvos():
    alvos = {}
    for linha in open(ALVOS_INFO):
        if linha.startswith('#') or not linha.strip():
            continue
        p = linha.split()
        if len(p) >= 2:
            alvos[p[0]] = p[1:]
    return alvos


def melhor_por_ligante(caminho, sinal):
    """Score de cada ligante = melhor entre as suas ~100 poses."""
    melhores = {}
    for linha in open(caminho):
        p = linha.split()
        if len(p) < 2 or linha.startswith('#'):
            continue
        try:
            v = sinal * float(p[-1])
        except ValueError:
            continue
        # '<ligante>_ligand_<n>' -> '<ligante>'
        lig = p[0].split('_')[0]
        if lig not in melhores or v > melhores[lig]:
            melhores[lig] = v
    return melhores


def metricas(resultados, alvos):
    n_lig = max((len(v) for v in resultados.values()), default=0)
    if not n_lig:
        return None
    cortes = {1: max(1, round(n_lig*0.01)), 5: max(1, round(n_lig*0.05)),
              10: max(1, round(n_lig*0.10))}
    suc = {k: [] for k in cortes}
    ef = {k: [] for k in cortes}
    for alvo, scores in resultados.items():
        ativos = [a for a in alvos.get(alvo, []) if a in scores]
        if not ativos:
            continue
        ordenado = sorted(scores, key=lambda l: -scores[l])
        for pct, n in cortes.items():
            topo = ordenado[:n]
            suc[pct].append(1.0 if ativos[0] in topo else 0.0)
            ef[pct].append(sum(1 for a in ativos if a in topo)
                           / (len(ativos)*pct*0.01))
    return {p: (100*np.mean(suc[p]), np.mean(ef[p])) for p in cortes}


def carregar_ref(nome, alvos):
    pasta = os.path.join(DIR_EX, nome)
    melhor = None
    for sinal in (1.0, -1.0):
        res = {}
        for alvo in alvos:
            c = os.path.join(pasta, f"{alvo}_score.dat")
            if os.path.exists(c):
                m = melhor_por_ligante(c, sinal)
                if m:
                    res[alvo] = m
        if not res:
            return None, "+"
        m = metricas(res, alvos)
        if m and (melhor is None or m[1][1] > melhor[0][1][1]):
            melhor = (m, "+" if sinal > 0 else "-")
    return melhor


def carregar_nosso(csv_path):
    res = {}
    for r in csv.DictReader(open(csv_path)):
        res.setdefault(r["alvo"], {})[r["ligante"]] = float(r["score"])
    return res


def main():
    nosso = sys.argv[1] if len(sys.argv) > 1 else None
    alvos = ler_alvos()
    linhas = []
    for nome in sorted(os.listdir(DIR_EX)):
        if not os.path.isdir(os.path.join(DIR_EX, nome)):
            continue
        r = carregar_ref(nome, alvos)
        if r and r[0]:
            linhas.append((nome, r[0], r[1]))
    if nosso and os.path.exists(nosso):
        m = metricas(carregar_nosso(nosso), alvos)
        if m:
            linhas.append((f"PhysScore ({os.path.basename(nosso)})", m, "+"))

    linhas.sort(key=lambda t: -t[1][1][1])   # ordena por EF1%
    print(f"\nScreening power — CASF-2016, recomputado com a mesma métrica\n")
    print(f"{'#':>3}  {'funcao':<34}{'suc1%':>7}{'EF1%':>7}{'suc5%':>7}{'EF5%':>7}"
          f"{'suc10%':>8}{'EF10%':>7}{'sinal':>7}")
    print("-" * 90)
    for i, (nome, m, sinal) in enumerate(linhas, 1):
        marca = "  <--" if "PhysScore" in nome else ""
        print(f"{i:>3}  {nome:<34}{m[1][0]:>7.1f}{m[1][1]:>7.2f}{m[5][0]:>7.1f}"
              f"{m[5][1]:>7.2f}{m[10][0]:>8.1f}{m[10][1]:>7.2f}{sinal:>7}{marca}")
    print("-" * 90)
    ref = [m[1][1] for n, m, _ in linhas if "PhysScore" not in n]
    print(f"     {'mediana das referencias (EF1%)':<34}{'':>7}{np.median(ref):>7.2f}")


if __name__ == "__main__":
    main()
