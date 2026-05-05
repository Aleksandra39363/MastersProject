#!/bin/bash
#
# Two-stage training for 20-agent HierLocal with IC3Net hard attention
# Stage 1: High entropy (0.0005) for exploration - 1500 epochs
# Stage 2: Zero entropy (0.0) for exploitation - 1500 epochs
#

set -e

# Disable CUDA, use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Configuration
SEEDS="${SEEDS:-1 2 3 4 5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1500}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1500}"
NPROCESSES="${NPROCESSES:-4}"
EPOCH_SIZE="${EPOCH_SIZE:-20}"
BATCH_SIZE="${BATCH_SIZE:-75}"

echo "=========================================="
echo "Stage 1: High Entropy Exploration"
echo "Epochs: $STAGE1_EPOCHS, Entropy: 0.0005"
echo "Epochs: $STAGE1_EPOCHS, Entropy: 0.0005"
echo "=========================================="

for s in $SEEDS; do
  echo ""
  echo "Seed $s - Stage 1 (Exploration with high entropy)"
  python3 main.py \
    --env_name traffic_junction \
    --nagents 20 \
    --add_rate_min 1.0 --add_rate_max 1.0 \
    --nprocesses $NPROCESSES \
    --epoch_size $EPOCH_SIZE \
    --batch_size $BATCH_SIZE \
    --hid_size 128 \
    --comm_passes 1 \
    --lrate 0.0001 \
    --dim 20 \
    --entr 0.0005 \
    --max_steps 500 \
    --ic3net --hard_attn --attn_heads 2 \
    --vision 2 \
    --difficulty medium \
    --crash_penalty -15 \
    --terminal_reward 15 \
    --num_epochs $STAGE1_EPOCHS \
    --seed $s \
    --hier_local_attn --team_size 5 \
    --save model_20agents_HierLocal_2stage_s${s}_stage1.pt \
    --log_path run_log_20agents_HierLocal_2stage_s${s}_stage1.pt
done

echo ""
echo "=========================================="
echo "Stage 2: Zero Entropy Fine-tuning"
echo "Epochs: $STAGE2_EPOCHS, Entropy: 0.0"
echo "Loading Stage 1 checkpoints"
echo "=========================================="

for s in $SEEDS; do
  echo ""
  echo "Seed $s - Stage 2 (Fine-tuning with zero entropy)"
  python3 main.py \
    --env_name traffic_junction \
    --nagents 20 \
    --add_rate_min 1.0 --add_rate_max 1.0 \
    --nprocesses $NPROCESSES \
    --epoch_size $EPOCH_SIZE \
    --batch_size $BATCH_SIZE \
    --hid_size 128 \
    --comm_passes 1 \
    --lrate 0.0001 \
    --dim 20 \
    --entr 0.0 \
    --max_steps 500 \
    --ic3net --hard_attn --attn_heads 2 \
    --vision 2 \
    --difficulty medium \
    --crash_penalty -15 \
    --terminal_reward 15 \
    --num_epochs $STAGE2_EPOCHS \
    --seed $s \
    --hier_local_attn --team_size 5 \
    --load model_20agents_HierLocal_2stage_s${s}_stage1.pt \
    --save model_20agents_HierLocal_2stage_s${s}.pt \
    --log_path run_log_20agents_HierLocal_2stage_s${s}.pt
done

echo ""
echo "=========================================="
echo "Two-stage training complete!"
echo "Final models: model_20agents_HierLocal_2stage_s*.pt"
echo "Final logs: run_log_20agents_HierLocal_2stage_s*.pt"
echo "=========================================="
