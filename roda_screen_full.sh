#!/bin/bash
# Screening power do nivel `full` (6 checkpoints). ~75 min por modo.
cd "$(dirname "$0")"
CK="exp_pose_fisica.pth exp_pose_fisica_s1.pth exp_pose_fisica_s2.pth \
    exp_direcional_fisica.pth exp_direcional_fisica_s1.pth exp_direcional_fisica_s2.pth"
echo "=== FULL · cabeca de POSE · $(date +%H:%M) ==="
python3 screening_power.py --jobs 14 --cabeca pose --pesos $CK --saida logs/screen_full_pose.csv
echo
echo "=== FULL · cabeca de AFINIDADE · $(date +%H:%M) ==="
python3 screening_power.py --jobs 14 --cabeca afinidade --pesos $CK --saida logs/screen_full_afin.csv
echo "=== fim: $(date +%H:%M) ==="
