#!/bin/bash
# Item 3: camadas direcionais. Dois treinos em sequência.
#
# O primeiro isola o efeito das features angulares — mesma configuração da linha de
# referência (R = 0,754 / 47,0% de top1), trocando apenas as camadas escalares por
# camadas com vetores equivariantes. É o que testa a hipótese: se a discriminação
# fina falha por falta de ângulo, este treino deve melhorar o slope E o docking
# power, sem nenhuma física de pose ajudando.
#
# O segundo combina com a física da seção 5.
cd "$(dirname "$0")"
echo "=== [1/2] direcional SEM física de pose — $(date +%H:%M) ==="
PHARM_DIRECIONAL=1 PHARM_CKPT=exp_direcional.pth python3 -u train.py
echo
echo "=== [2/2] direcional COM física de pose — $(date +%H:%M) ==="
PHARM_DIRECIONAL=1 PHARM_POSE=1 PHARM_CKPT=exp_direcional_fisica.pth python3 -u train.py
echo
echo "=== fim: $(date +%H:%M) ==="
