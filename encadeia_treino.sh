#!/bin/bash
# Espera o process() do dataset terminar e, SÓ se ele tiver concluído com sucesso,
# emenda o treino. A verificação não é burocracia: se o processamento cair (queda de
# energia, OOM), o `.pt` não existe e treinar em cima do dataset anterior — ou
# estourar num arquivo ausente — desperdiçaria horas sem ninguém olhando.
cd /home/lucas/Documents/PINN/PharmXAI-3D
PID_PROC=33319
LOG_PROC=logs/process_20260903_0938.log
LOG_TREINO=logs/train_$(date +%Y%m%d_%H%M).log

while ps -p "$PID_PROC" > /dev/null 2>&1; do sleep 30; done

if [ ! -f processed/pharm_xai_dataset.pt ]; then
    echo "[X] processed/pharm_xai_dataset.pt nao existe. Treino NAO iniciado." | tee -a "$LOG_TREINO"
    echo "    Ultimas linhas do processamento:" | tee -a "$LOG_TREINO"
    tail -5 "$LOG_PROC" | tee -a "$LOG_TREINO"
    exit 1
fi
if ! grep -q "Dataset salvo" "$LOG_PROC"; then
    echo "[X] O log nao registra 'Dataset salvo'. Treino NAO iniciado." | tee -a "$LOG_TREINO"
    tail -5 "$LOG_PROC" | tee -a "$LOG_TREINO"
    exit 1
fi

echo "[+] Dataset pronto ($(date +%H:%M)). Iniciando o treino." | tee -a "$LOG_TREINO"
python -u train.py >> "$LOG_TREINO" 2>&1
echo "[+] Treino encerrado com codigo $? em $(date +%H:%M)." | tee -a "$LOG_TREINO"
