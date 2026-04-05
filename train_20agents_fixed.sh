#!/bin/bash
# Fixed training for 20 agents with IC3Net

echo "=== Training 20 Agents with IC3Net (FIXED) ==="
echo "Key changes:"
echo "  - Entropy: 0.001 (enables exploration)"
echo "  - Hidden size: 256 (better attention capacity)"  
echo "  - Comm passes: 3 (info propagation)"
echo "  - Learning rate: 0.0005 (stability)"
echo ""

# Recommended: Use CommNet (NOT attention) for 20 agents
# Attention is optimized for 10 agents, fails to scale past that

python3 main.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 20 \
  --hid_size 256 \
  --comm_passes 3 \
  --entr 0.001 \
  --lrate 0.0005 \
  --num_epochs 1500 \
  --epoch_size 10 \
  --batch_size 500 \
  --nprocesses 16 \
  --max_steps 20 \
  --seed 1 \
  --eval_every 100 \
  --eval_epochs 10 \
  --save model_20agents_fixed_seed1.pt \
  --log_path run_log_20agents_fixed_seed1.pt

# If above STILL struggles, also try:
# --recurrent (adds LSTM for temporal memory)
# --normalize_rewards (helps with varying agent rewards)

echo ""
echo "Training complete! Compare with:"
echo "  python3 visualize_training.py --plot-training --log_file run_log_20agents_fixed_seed1.pt"
