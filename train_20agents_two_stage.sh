#!/bin/bash
set -e

# Two-stage training for 20 agents:
# Stage 1: tiny entropy for stability
# Stage 2: entropy=0 fine-tuning from Stage 1 checkpoint

COMMON="--env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 \
--nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 \
--lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium \
--crash_penalty -15 --terminal_reward 15"

# Stage lengths
STAGE1_EPOCHS=800
STAGE2_EPOCHS=1200

for s in 1 2 3; do
  echo "=== Seed ${s}: Baseline Stage 1 (entr=5e-5) ==="
  python3 main.py $COMMON --num_epochs ${STAGE1_EPOCHS} --entr 0.00005 --seed $s \
    --save model_20agents_Base_s${s}_stage1.pt \
    --log_path run_log_20agents_Base_s${s}_stage1.pt

  echo "=== Seed ${s}: Baseline Stage 2 (entr=0, from Stage 1) ==="
  python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
    --load model_20agents_Base_s${s}_stage1.pt \
    --save model_20agents_Base_s${s}_stage2.pt \
    --log_path run_log_20agents_Base_s${s}_stage2.pt

done

for s in 1 2 3; do
  echo "=== Seed ${s}: Flash Stage 1 (entr=5e-5) ==="
  python3 main.py $COMMON --num_epochs ${STAGE1_EPOCHS} --entr 0.00005 --seed $s \
    --flash --attn_heads 4 \
    --save model_20agents_Flash_s${s}_stage1.pt \
    --log_path run_log_20agents_Flash_s${s}_stage1.pt

  echo "=== Seed ${s}: Flash Stage 2 (entr=0, from Stage 1) ==="
  python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
    --flash --attn_heads 4 \
    --load model_20agents_Flash_s${s}_stage1.pt \
    --save model_20agents_Flash_s${s}_stage2.pt \
    --log_path run_log_20agents_Flash_s${s}_stage2.pt

done

for s in 1 2 3; do
  echo "=== Seed ${s}: Mamba Stage 1 (entr=5e-5) ==="
  python3 main.py $COMMON --num_epochs ${STAGE1_EPOCHS} --entr 0.00005 --seed $s \
    --mamba \
    --save model_20agents_Mamba_s${s}_stage1.pt \
    --log_path run_log_20agents_Mamba_s${s}_stage1.pt

  echo "=== Seed ${s}: Mamba Stage 2 (entr=0, from Stage 1) ==="
  python3 main.py $COMMON --num_epochs ${STAGE2_EPOCHS} --entr 0.0 --seed $s \
    --mamba \
    --load model_20agents_Mamba_s${s}_stage1.pt \
    --save model_20agents_Mamba_s${s}_stage2.pt \
    --log_path run_log_20agents_Mamba_s${s}_stage2.pt

done

echo "=== Two-stage runs complete ==="
