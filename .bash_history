python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed3.pt --agent_counts 20 --smooth 100
tmux
tmux list-panes -a
tmux ls
tmux attach -t 3
tmux attach -t 4
tmux attach -t 5
tmux attach -t 5
tmux attach -t 0
tmux attach -t 1
tmux ls
tmux
tmux attach -t 2
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
conda activate thesis
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --comm_passes 2 --entr 0.0005 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_5mediumagents_Atnseed${s}.pt --log_path run_log_5mediumagents_Atnseed${s}.pt; done
q
tmux attach -t 1
tmux kill-session -t 1
tmux kill-session -t 0
tmux ls
tmux attach -t 2
tmux list-panes -a
tmux attach -t 5
tmux
tmux kill-session -t 5
tmux attach -t 0
tmux attach -t 1
tmux attach -t 5
tmux attach -t 1
tmux ls
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed4.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed3.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed2.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed1.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed3.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed4.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed2.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed1.pt --agent_counts 5
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed2.pt --agent_counts 5
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed3.pt --agent_counts 5
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed4.pt --agent_counts 5
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed4.pt --agent_counts 5 --smooth 100
tmux ls
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 500 --hid_size 128 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_20mediumagents_Atnseed${s}.pt --log_path run_log_20mediumagents_Atnseed${s}.pt; done
conda activate thesis
tmux
tmux ls
conda activate thesis
tmux
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed2.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed1.pt run_log_5mediumagents_Atnseed2.pt run_log_5mediumagents_Atnseed3.pt run_log_5mediumagents_Atnseed4.pt run_log_5mediumagents_Atnseed5.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed1.pt run_log_5mediumagents_Atnseed2.pt run_log_5mediumagents_Atnseed3.pt run_log_5mediumagents_Atnseed4.pt run_log_5mediumagents_Atnseed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt run_log_10agents_Atnseed2.pt run_log_10agents_Atnseed3.pt run_log_10agents_Atnseed4.pt run_log_10agents_Atnseed5.pt --agent_counts 10 10 10 10 10 --smooth 100
conda activate thesis
tmux ls
tmux ls
tmux
git add .
git commit -m "added flash attention and mamba"
git push
conda activate thesis
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --flash --attn_heads 4 --save model_10agents_FlashAtnseed${s}.pt --log_path run_log_10agents_FlashAtnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --flash --attn_heads 4 --save model_10agents_FlashAtnseed${s}.pt --log_path run_log_10agents_FlashAtnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --mamba --save model_10agents_Mambaseed${s}.pt --log_path run_log_10agents_Mambaseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --mamba --save model_10agents_Mambaseed${s}.pt --log_path run_log_10agents_Mambaseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 500 --hid_size 128 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_20mediumagents_Atnseed${s}.pt --log_path run_log_20mediumagents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 500 --hid_size 128 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_FlashAtnseed1.pt --agent_counts 10 --smooth 100
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_FlashAtnseed1.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_FlashAtnseed1.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Mambaseed1.pt --agent_counts 10
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
tmux ls
tmux kill -t 10
tmux kill-session -t 10
tmux kill-session -t 11
tmux ls
tmux attach -t 7
tmux attach -t 8
tmux
git add ,
git add .
git commit -m "trying enthopy"
git push
tmux ls
tmux attach -t 9
tmux kill-session -t 9
tmux kill-session -t 7
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
tmux
tmux
chmod +x /tmp/train_20agents_comparison.sh
cat /tmp/train_20agents_comparison.sh
chmod +x /tmp/train_20agents_two_stage.sh
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt  run_log_5mediumagents_seed2.pt run_log_5mediumagents_seed3.pt run_log_5mediumagents_seed4.pt run_log_5mediumagents_seed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_Atnseed1.pt  run_log_5mediumagents_Atnseed2.pt run_log_5mediumagents_Atnseed3.pt run_log_5mediumagents_Atnseed4.pt run_log_5mediumagents_Atnseed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt  run_log_10agents_Atnseed2.pt run_log_10agents_Atnseed3.pt run_log_10agents_Atnseed4.pt  --agent_counts 10 10 10 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed1.pt  run_log_10agents_seed2.pt run_log_10agents_seed3.pt run_log_10agents_seed4.pt run_log_10agents_seed5.pt --agent_counts 10 10 10 10 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt run_log_20mediumagents_seed2-sporo.pt --agent_counts 20 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s1_stage2.pt --agent_counts 20 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s1_stage2.pt --agent_counts 20 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage2.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt  run_log_10agents_Atnseed2.pt run_log_10agents_Atnseed3.pt run_log_10agents_Atnseed4.pt  --agent_counts 10 10 10 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage2.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
conda activate thesis
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt -agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage2.pt run_log_20agents_Base_s2_stage2.pt run_log_20agents_Base_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1_stage1.pt run_log_20agents_Flash_s2_stage1.pt run_log_20agents_Flash_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1_stage2.pt run_log_20agents_Flash_s2_stage2.pt run_log_20agents_Flash_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage1.pt run_log_20agents_Mamba_s2_stage1.pt run_log_20agents_Mamba_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage2.pt run_log_20agents_Mamba_s2_stage2.pt run_log_20agents_Mamba_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage1.pt run_log_20agents_Mamba_s2_stage1.pt run_log_20agents_Mamba_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage2.pt run_log_20agents_Mamba_s2_stage2.pt run_log_20agents_Mamba_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage1.pt run_log_20agents_Mamba_s2_stage1.pt run_log_20agents_Mamba_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1_stage2.pt run_log_20agents_Mamba_s2_stage2.pt run_log_20agents_Mamba_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt run_log_20mediumagents_seed2.pt run_log_20mediumagents_seed3.pt run_log_20mediumagents_seed4.pt run_log_20mediumagents_seed5.pt --agent_counts 20 20 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt run_log_20mediumagents_seed2.pt run_log_20mediumagents_seed3.pt run_log_20mediumagents_seed4.pt run_log_20mediumagents_seed5.pt --agent_counts 20 20 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage2.pt run_log_20agents_Base_s2_stage2.pt run_log_20agents_Base_s3_stage2.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt run_log_20mediumagents_seed2.pt run_log_20mediumagents_seed3.pt run_log_20mediumagents_seed4.pt run_log_20mediumagents_seed5.pt --agent_counts 20 20 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
conda activate thesis 
pip install mamba-ssm -q
tmux
conda activate thesis
tmux
conda activate thesis
tmux
conda activate thesis
tmux
conda activate thesis
tmux
git add .
git commit -m "changed mamba implementation and added trained files"
git push
git pull
git branch
git pull --rebase origin master
git add .
 git rebase --continue
git push origin master
conda activate thesis
tmux
conda activate thesis
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 500 --hid_size 128 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_5mediumagents_Atnseed${s}.pt --log_path run_log_5mediumagents_Atnseed${s}.pt; done
tmuxq
q
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_10agents_Atnseed${s}.pt --log_path run_log_10agents_Atnseed${s}.pt; done
for s in 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
bash train_20agents_two_stage.sh
bash train_20agents_two_stage.sh
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --flash --attn_heads 4 --save model_20agents_Flash_s${s}.pt --log_path run_log_20agents_Flash_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 30 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 24 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --flash --attn_heads 4 --save model_30agents_Flash_s${s}.pt --log_path run_log_30agents_Flash_s${s}.pt; done
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt run_log_20agents_Flash_s1.pt run_log_20agents_Mamba_s1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20 --smooth 100
tmux
tmux ls
tmux attach -t 8
tmux attach -t 8
tmux kill-session -t 8
tmux ls
tmux kill-session -t 3
tmux kill-session -t 4
tmux ls
tmux kill-session -t 2
tmux kill-session -t 12
tmux kill-session -t 13
tmux kill-session -t 14
tmux ls
tmux attach -t 17
tmux attach -t 19
tmux kill-session -t 19
tmux attach -t 20
tmux kill-session -t 20
tmux
conda activate thesis
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt --agent_counts 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1_final.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt --agent_counts 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt --agent_counts 30 
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 24 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --flash --attn_heads 4 --save model_30agents_Flash_s${s}.pt --log_path run_log_30agents_Flash_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 24 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --flash --attn_heads 4 --save model_30agents_Flash_s${s}.pt --log_path run_log_30agents_Flash_s${s}.pt; done
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_stage1.pt run_log_20agents_Base_s2_stage1.pt run_log_20agents_Base_s3_stage1.pt --agent_counts 20 20 20 --smooth 100
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s -- mamba --mamba_dropout 0.1 --max_grad_norm 1.0 --save model_20agents_Mamba_s${s}.pt --log_path run_log_20agents_Mamba_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s -- ma-mamba --mamba_dropout 0.1 --max_grad_norm 1.0 --save model_20agents_Mamba_s${s}.pt --log_path run_log_20agents_Mamba_s${s}.pt; done[D
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt --agent_counts 30 
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt run_log_25agents_Base_s2.pt --agent_counts 25 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt run_log_30agents_Base_s1.pt --agent_counts 30 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt run_log_30agents_Base_s2.pt --agent_counts 30 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt run_log_25agents_Base_s2.pt --agent_counts 25 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Flash_s1.pt run_log_30agents_Flash_s2.pt --agent_counts 30 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt run_log_20agents_Mamba_s2.pt --agent_counts 20 20 --smooth 100
tmux ls
tmux attach -t 15
tmux attach -t 16
tmux attach -t 17
tmux attach -t 18
tmux attach -t 21
tmux attach -t 15
tmux attach -t 22
tmux attach -t 23
tmux attach -t 24
tmux attach -t 25
tmux kill-session -t 25
tmux kill-session -t 24
tmux kill-session -t 23
tmux kill-session -t 21
tmux kill-session -t 15
tmux ls
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Flash_s2.pt --agent_counts 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Flash_s1.pt --agent_counts 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s2.pt --agent_counts 30 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s2.pt --agent_counts 25 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s2.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s2_final.pt --agent_counts 20 --smooth 100
tmux
tmux
wc -l logs/optuna_hierlocal/trial_* | tail -1
ps aux | grep main.py | grep optuna   
conda activate thesis
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test.pt --agent_counts 20 --smooth 100
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test.pt --agent_counts 20 --smooth 1000
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test_s1.pt --agent_counts 20 
conda activate thesis
tmux
tmux
tmux
pip install -r requirements.txt
conda activate thesis
pip install -r requirements.txt
tmux
conda activate thesis
python3 scripts/sweep_team_size.py --dry-run
python3 scripts/sweep_team_size.py --team-sizes 5 --seeds 1 --dry-run
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 100 --entr 0.00005 --seed 1 --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test.pt --log_path run_log_20agents_HierLocal_test.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 100 --entr 0.00005 --seed 1 --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test.pt --log_path run_log_20agents_HierLocal_test.pt
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 500 --ic3net --vision 2 --comm_passes 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test_s${s}.pt --log_path run_log_20agents_HierLocal_test_s${s}.pt
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 500 --ic3net --vision 2 --comm_passes 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test_s${s}.pt --log_path run_log_20agents_HierLocal_test_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 500 --ic3net --vision 2 --comm_passes 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test_s${s}.pt --log_path run_log_20agents_HierLocal_test_s${s}.pt;  for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 500 --ic3net --vision 2 --comm_passes 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test_s${s}.pt --log_path run_log_20agents_HierLocal_test_s${s}.pt; 
bash /home/dragojla/scripts/fine_tune_20agents_hierlocal.sh
python3 scripts/sweep_team_size.py --team-sizes 4 6 8 10 --seeds 1 2 3
python3 scripts/sweep_team_size.py --team-sizes 4 6 8 10 --seeds 1 2 3
python3 scripts/sweep_team_size.py --team-sizes 4 6 8 10 --seeds 1 2 3
File "/home/dragojla/main.py", line 875, in <module>
    main()   File "/home/dragojla/main.py", line 861, in main
    run(args.num_epochs)
  File "/home/dragojla/main.py", line 667, in run
    s = trainer.train_batch(ep)
        ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/trainer.py", line 275, in train_batch
    batch, stat = self.run_batch(epoch)
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/trainer.py", line 263, in run_batch
    episode, episode_stat = self.get_episode(epoch)
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/trainer.py", line 65, in get_episode
    action_out, value = self.policy_net(x, info)
                        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/comm_attention.py", line 265, in forward
    global_comm, _ = self.global_attention(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/modules/activation.py", line 1266, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/functional.py", line 5291, in multi_head_attention_forward
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dragojla/miniconda3/envs/thesis/lib/python3.12/site-packages/torch/nn/functional.py", line 5115, in _mha_shape_check
    raise AssertionError(
AssertionError: query should be unbatched 2D or batched 3D tensor but received 4-D query tensor
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 100 --entr 0.00005 --seed 1 --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test.pt --log_path run_log_20agents_HierLocal_test.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 100 --entr 0.00005 --seed 1 --hier_local_attn --team_size 5 --comm_passes 2 --save model_20agents_HierLocal_test.pt --log_path run_log_20agents_HierLocal_test.pt
conda activate thesis
tmux
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 100 --entr 0.00005 --seed 1 --hier_local_attn --team_size 5 --save model_20agents_HierLocal_test_s1.pt --log_path run_log_20agents_HierLocal_test_s1.pt
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test_s1.pt --agent_counts 20 
tmux ls
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test_s1.pt --agent_counts 20 
tmux ls
tmux attach -t 32
tmux attach -t 31
tmux attach -t 30
tmux attach -t 29
tmux attach -t 28
tmux attach -t 27
tmux attach -t 26
tmux kill-session -t 32
tmux kill-session -t 31
tmux kill-session -t 29
tmux kill-session -t 28
tmux kill-session -t 27
tmux kill-session -t 26
tmux ls
tmux attach -t 30
conda activate thesis
tmux
for s in 1 2 3; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --epoch_size 20 --batch_size 150 --hid_size 256 --comm_passes 2 --lrate 0.0001 --dim 20 --entr 0.00005 --max_steps 500 --ic3net --hard_attn --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --seed $s --flash --save model_20agents_Flash_s${s}.pt --log_path run_log_20agents_Flash_s${s}.pt; done
for s in 1 2 3; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --epoch_size 20 --batch_size 150 --hid_size 256 --comm_passes 2 --lrate 0.0001 --dim 20 --entr 0.00005 --max_steps 500 --ic3net --hard_attn --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --seed $s --flash --save model_20agents_Flash_s${s}.pt --log_path run_log_20agents_Flash_s${s}.pt; done
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 
conda activate thesis
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 
tmux ls
tmux attach -t 34
tmux attach -t 33
tmux kill-session -t 34
tmux
conda activate thesis
for s in 1 2 3; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --epoch_size 20 --batch_size 150 --hid_size 256 --comm_passes 2 --lrate 0.0001 --dim 20 --entr 0.00005 --max_steps 500 --ic3net --hard_attn --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --seed $s --flash --save model_20agents_Flash_s${s}.pt --log_path run_log_20agents_Flash_s${s}.pt; done
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_2stage_s1_stage1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_test.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
for s in 1 2 3; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --epoch_size 20 --batch_size 150 --hid_size 256 --comm_passes 2 --lrate 0.0001 --dim 20 --entr 0.00005 --max_steps 500 --ic3net --hard_attn --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --seed $s --flash --save model_20agents_Flash_s${s}.pt --log_path run_log_20agents_Flash_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --mamba --mamba_dropout 0.1 --max_grad_norm 1.0 --save model_20agents_Mamba_s${s}.pt --log_path run_log_20agents_Mamba_s${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs ${STAGE1_EPOCHS} --entr 0.00005 --seed $s --save model_20agents_Base_s${s}_final.pt --log_path run_log_20agents_Base_s${s}_final.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --save model_20agents_Base_s${s}_final.pt --log_path run_log_20agents_Base_s${s}_final.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 30 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 24 --max_steps 700 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --entr 0.00005 --seed $s --save model_30agents_Base_s${s}.pt --log_path run_log_30agents_Base_s${s}.pt; done
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_2stage_s1_stage1.pt --agent_counts 20 
tmux ls
tmux attach -t 16
tmux attach -t 17
tmux attach -t 16
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s1_final.pt run_log_20agents_Base_s2_final.pt --agent_counts 20 20
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s2.pt run_log_25agents_Base_s1.pt --agent_counts 25 25
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s2.pt run_log_25agents_Base_s1.pt run_log_25agents_Base_s3.pt --agent_counts 25 25 25
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s2.pt run_log_30agents_Base_s1.pt run_log_30agents_Base_s2.pt --agent_counts 30 30 
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt run_log_30agents_Base_s2.pt --agent_counts 30 30
tmux ls
tmux attach -t 22
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20
tmux ls
tmux attach -t 30
tmux attach -t 30
tmux attach -t 33
tmux attach -t 35
tmux attach -t 36
tmux attach -t 37
tmux attach -t 36
tmux attach -t 35
tmux kill-session -t 35
tmux ls
tmux ls
tmux attach -t 37
tmux attach -t 36
tmux attach -t 33
tmux attach -t 30
tmux attach -t 22
tmux kill-session -t 22
tmux attach -t 16
tmux attach -t 17
tmux attach -t 18
tmux kill-session -t 16
 python3 visualize_training.py --compare-scaling --log_files run_log_30agents_Base_s1.pt run_log_30agents_Base_s2.pt --agent_counts 30 30
tmux attach -t 18
tmux kill-session -t 18
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s3.pt --agent_counts 25
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_2stage_s1_stage1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_HierLocal_s1.pt --agent_counts 20 
tmux ls
tmux attach -t 17
tmux attach -t 30
tmux attach -t 33
tmux attach -t 36
tmux attach -t 37
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s1.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s2.pt --agent_counts 20 
print('-' * 50)
for n in [10, 25, 50, 100, 200, 500]:;     n2, flash, red = memory_comparison(n)
print() print('Key Benefits of Flash Attention:')
print('• O(n) memory complexity vs O(n²)')
print('• Scales to 1000+ agents efficiently') 
print('• Faster training and inference')
print('• Exact attention (not approximated)')
"
tmux ls
tmux attach -t 17
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25 
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt  run_log_25agents_Base_s2.pt --agent_counts 25 25
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt  run_log_25agents_Base_s3.pt --agent_counts 25 25
 python3 visualize_training.py --compare-scaling --log_files model_1.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files model_2agents-deletecrashedcars3.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files model_2agents-es.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files model_5agents-NoAtn3easy2.pt --agent_counts 5
 python3 visualize_training.py --compare-scaling --log_files model_5agents-NoAtn3medium2.pt --agent_counts 5
 python3 visualize_training.py --compare-scaling --log_files model_5agents.pt --agent_counts 5
 python3 visualize_training.py --compare-scaling --log_files model_5easyagents_seed1.pt --agent_counts 5
 python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files run_log_2agents-es.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Mamba_s2.pt --agent_counts 20 
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s1.pt --agent_counts 25
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Flash_s1.pt --agent_counts 20
 python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt run_log_20mediumagents_seed2.pt run_log_20mediumagents_seed3.pt run_log_20mediumagents_seed4.pt run_log_20mediumagents_seed5.pt --agent_counts 20 20 20 20 20
 python3 visualize_training.py --compare-scaling --log_files  run_log_25agents_Base_s3.pt run_log_25agents_Base_s3.pt run_log_25agents_Base_s3.pt --agent_counts 25 25 25
 python3 visualize_training.py --compare-scaling --log_files  run_log_25agents_Base_s3.pt run_log_25agents_Base_s3.pt --agent_counts 25 25 25
 python3 visualize_training.py --compare-scaling --log_files  run_log_25agents_Base_s3.pt run_log_25agents_Base_s3.pt --agent_counts 25 25 
conda activate thesis
tmux
tmux ls
tmux attach -t 39
tmux attach -t 38
tmux attach -t 39
tmux kill-session -t 39
for s in 1 2 3; do python3 main.py --env_name traffic_junction --nagents 50 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --epoch_size 20 --batch_size 150 --hid_size 256 --comm_passes 2 --lrate 0.0001 --dim 40 --entr 0.001 --max_steps 100 --ic3net --hard_attn --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --num_epochs 3000 --seed $s --flash --save model_50agents_FlashNew_s${s}.pt --log_path run_log_50agents_FlashNew_s${s}.pt; done
conda activate thesis
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_FlashNew_s1.pt --agent_counts 20
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_Base_s3.pt --agent_counts 25
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_Base_s3.pt --agent_counts 25
conda activate thesis
tmux
tmux ls
tmux attach -t 38
tmux attach -t 37
tmux attach -t 36
tmux attach -t 33
tmux attach -t 30
tmux attach -t 17
tmux
 python3 visualize_training.py --compare-scaling --log_files run_log_50agents_FlashNew_s1.pt --agent_counts 50
 python3 visualize_training.py --compare-scaling --log_files run_log_25agents_FlashNew_s1.pt --agent_counts 25
 python3 visualize_training.py --compare-scaling --log_files run_log_20agents_FlashNew_s1.pt --agent_counts 20
