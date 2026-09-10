#!/bin/bash
# Sementes pareadas para o item 3 (seção 6).
#
# A semente 0 das duas configurações já existe: exp_pose_fisica.pth (escalar,
# R = 0,750) e exp_direcional_fisica.pth (direcional, R = 0,768). Faltam 1 e 2.
#
# Pareado importa: a mesma semente dá a mesma inicialização e a mesma ordem de
# lotes nas duas configurações, de modo que a diferença entre elas é a
# configuração e não o sorteio. Na seção 3 isso reduziu o desvio de ±0,016 para
# ±0,006 — a diferença entre medir um efeito e não conseguir medi-lo.
cd "$(dirname "$0")"
for s in 1 2; do
  echo "=== escalar + fisica · semente $s · $(date +%H:%M) ==="
  PHARM_POSE=1 PHARM_SEED=$s PHARM_CKPT=exp_pose_fisica_s$s.pth python3 -u train.py
  echo
  echo "=== direcional + fisica · semente $s · $(date +%H:%M) ==="
  PHARM_DIRECIONAL=1 PHARM_POSE=1 PHARM_SEED=$s PHARM_CKPT=exp_direcional_fisica_s$s.pth python3 -u train.py
  echo
done
echo "=== fim: $(date +%H:%M) ==="
