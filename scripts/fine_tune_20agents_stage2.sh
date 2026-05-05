#!/bin/bash
set -e

# Fine-tune (Stage 2) script for 20-agent experiments.
# Usage: ./scripts/fine_tune_20agents_stage2.sh [seeds...]
# If no seeds provided, defaults to: 1 2 3

SEEDS="$@"
if [ -z "$SEEDS" ]; then
  SEEDS="1 2 3"
fi

# Common training args (match your baseline Stage 1)
COMMON="--env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 \
--nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 \
--lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium \
--crash_penalty -15 --terminal_reward 15"

STAGE2_EPOCHS=${STAGE2_EPOCHS:-1200}

for s in $SEEDS; do
  echo "=== Seed ${s}: Base Stage 2 (entr=0, from Stage 1) ==="
  if [ -f model_20agents_Base_s${s}_stage1.pt ]; then
    python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
      --load model_20agents_Base_s${s}_stage1.pt \
      --save model_20agents_Base_s${s}_stage2.pt \
      --log_path run_log_20agents_Base_s${s}_stage2.pt
  else
    echo "Warning: model_20agents_Base_s${s}_stage1.pt not found — skipping Base fine-tune for seed ${s}"
  fi

  echo "=== Seed ${s}: Flash Stage 2 (entr=0, from Stage 1) ==="
  if [ -f model_20agents_Flash_s${s}_stage1.pt ]; then
    python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
      --flash --attn_heads 4 \
      --load model_20agents_Flash_s${s}_stage1.pt \
      --save model_20agents_Flash_s${s}_stage2.pt \
      --log_path run_log_20agents_Flash_s${s}_stage2.pt
  else
    echo "Warning: model_20agents_Flash_s${s}_stage1.pt not found — skipping Flash fine-tune for seed ${s}"
  fi

  echo "=== Seed ${s}: Mamba Stage 2 (entr=0, from Stage 1) ==="
  if [ -f model_20agents_Mamba_s${s}_stage1.pt ]; then
    python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
      --mamba \
      --load model_20agents_Mamba_s${s}_stage1.pt \
      --save model_20agents_Mamba_s${s}_stage2.pt \
      --log_path run_log_20agents_Mamba_s${s}_stage2.pt
  else
    echo "Warning: model_20agents_Mamba_s${s}_stage1.pt not found — skipping Mamba fine-tune for seed ${s}"
  fi

done

echo "=== Fine-tune (Stage 2) runs complete ==="
