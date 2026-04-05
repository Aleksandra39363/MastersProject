python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --curr_start 0 --curr_end 400 --nprocesses 1 --num_epochs 1000 --epoch_size 3 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --early_stop --early_stop_threshold 0.95 --early_stop_patience 50 --save model_5agents-es.pt --log_path run_log_5agents-es.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --curr_start 0 --curr_end 400 --nprocesses 1 --num_epochs 1000 --epoch_size 3 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --early_stop --early_stop_threshold 0.95 --early_stop_patience 50 --save model_5agents-es.pt --log_path run_log_5agents-es.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --curr_start 0 --curr_end 400 --nprocesses 1 --num_epochs 1000 --epoch_size 3 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --early_stop --early_stop_threshold 0.95 --early_stop_patience 50 --save model_5agents-es.pt --log_path run_log_5agents-es.pt
python visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20 --smooth 100
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 14 --max_steps 150 --ic3net --vision 2 --difficulty medium --save model_10agents-nones.pt --log_path run_log_10agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 14 --max_steps 150 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 10 --save model_10agents-nones.pt --log_path run_log_10agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 14 --max_steps 150 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_10agents-nones.pt --log_path run_log_10agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-es.pt --agent_counts 10 --smooth 100
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 4000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 18 --max_steps 250 --ic3net --vision 2 --difficulty hard --save model_20agents-nones.pt --log_path run_log_20agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 4000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 18 --max_steps 250 --ic3net --vision 2 --difficulty hard --crash_penalty -5 --terminal_reward 10 --save model_20agents-nones.pt --log_path run_log_20agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 4000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 18 --max_steps 250 --ic3net --vision 2 --difficulty hard --crash_penalty -20 --terminal_reward 15 --save model_20agents-nones.pt --log_path run_log_20agents-nones.pt
git log
git diff 029e42a69a6dd31e119cad9ec845a8add63a22b8  02dc2f430599387e35e208d1c64f1730481de858
it branch
git branch
git diff
git status
git add .
git log
git log
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-nones.pt --agent_counts 10 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 10 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt run_log_10agents-nones.pt run_log_20agents-nones.pt --agent_counts 5 10 20 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_5agents-nones.pt --log_path run_log_5agents-nones.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt run_log_10agents-es.pt run_log_20agents-es.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files model_5agents-nones-withoutSuccessRewardBigStepPenalty.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-nones.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-nonescollless.pt --log_path run_log_5agents-nonescollless.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 1 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-nonescollless.pt --log_path run_log_5agents-nonescollless.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-nonescollless.pt --log_path run_log_5agents-nonescollless.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-es.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 15 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 15 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 20 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 20 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -5 --terminal_reward 10 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -20 --terminal_reward 10 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 10 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 10 --save model_5agents-deletecrashedcars.pt --log_path run_log_5agents-deletecrashedcars.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 100 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 10 --save model_5agents-deletecrashedcars2.pt --log_path run_log_5agents-deletecrashedcars2.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars2.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 80 --ic3net --vision 2 --difficulty easy --crash_penalty -5 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -5 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 10 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 30 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 30 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 30 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 2 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 120 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3.pt --log_path run_log_5agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 5000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 18 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --save model_20agents-deletecrashedcars3hard.pt --log_path run_log_20agents-deletecrashedcars3hard.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 4000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 18 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_20agents-deletecrashedcars3medium.pt --log_path run_log_20agents-deletecrashedcars3medium.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_10agents-deletecrashedcars3.pt --log_path run_log_10agents-deletecrashedcars3.pt
python3 visualize_training.py --compare-scaling --log_files run_log__5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-deletecrashedcars3medium.pt --agent_counts 5 --smooth 1
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-deletecrashedcars3.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-deletecrashedcars3.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents-deletecrashedcars3easy.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents-deletecrashedcars3medium.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents-deletecrashedcars3hard.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents-deletecrashedcars3.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents-deletecrashedcars3medium.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20agents-deletecrashedcars3hard.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 10
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3medium.pt --log_path run_log_5agents-deletecrashedcars3medium.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_5agents-deletecrashedcars3medium.pt --agent_counts 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3medium.pt --log_path run_log_5agents-deletecrashedcars3medium.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 3 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3medium.pt --log_path run_log_5agents-deletecrashedcars3medium.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_10agents-deletecrashedcars3.pt --log_path run_log_10agents-deletecrashedcars3.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 3 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_10agents-deletecrashedcars3.pt --log_path run_log_10agents-deletecrashedcars3.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 2 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 2 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy.pt --log_path run_log_5agents-deletecrashedcars3easy.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-NoAtn3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-NoAtn3easy2.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
git add .
git commit -m "working without attention"
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3medium2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 5 10 20 --smooth 100
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --use_agent_attn --attn_heads 4  --save model_10agents-WithAtn3-2.pt --log_path run_log_10agents-WithAtn3-2.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3medium2.pt --log_path run_log_5agents-deletecrashedcars3medium2.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 18 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --save model_20agents-deletecrashedcars3hard2.pt --log_path run_log_20agents-deletecrashedcars3hard2.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-deletecrashedcars3easy2.pt --log_path run_log_5agents-deletecrashedcars3easy2.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 use_agent_attn --attn_heads 4  --save model_5agents-deletecrashedcars3easy2.pt --log_path run_log_5agents-deletecrashedcars3easy2.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --use_agent_attn --attn_heads 4  --save model_5agents-deletecrashedcars3easy2.pt --log_path run_log_5agents-deletecrashedcars3easy2.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
git log --oneline
git reset --hard a48b7b9
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --use_agent_attn --attn_heads 4  --save model_5agents-WithAtn3easy2.pt --log_path run_log_5agents-WithAtn3easy2.pt
git log --oneline
git reset --hard a48b7b9
git reset --hard a48b7b9
git reset --hard a48b7b9
git checkout HEAD
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
git reset --hard a48b7b9
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --use_agent_attn --attn_heads 4  --save model_5agents-WithAtn3easy2.pt --log_path run_log_5agents-WithAtn3easy2.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt --agent_counts 5 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 20
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt --agent_counts 2 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-deletecrashedcars3easy.pt run_log_10agents-deletecrashedcars3.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 5 10 20 --smooth 100
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_10agents-deletecrashedcars3-2.pt --log_path run_log_10agents-deletecrashedcars3-2.pt
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-WithAtn3easy2.pt run_log_10agents-WithAtn3-2.pt --agent_counts 5 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-WithAtn3easy2.pt run_log_10agents-WithAtn3-2.pt --agent_counts 5 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-WithAtn3easy2.pt run_log_10agents-WithAtn3-2.pt --agent_counts 5 10 --smooth 100
tmux
tmux
tmux
tmux
tmux
tmux
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_5agents-15.pt run_log_10agents-15.pt run_log_20agents-15medium.pt --agent_counts 5 10 20 --smooth 100
git commit -m "working version, 0 penalty, +15 reward"
git add .
git commit -m "working version, 0 penalty, +15 reward"
tmux
git add .
git commit -m "changed visualisatio"
git add .
git commit 'm "added evaluation"

git commit -m "added evaluation"

git add .
git commit -m "added evaluation"
git status
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
tmux
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5mediumagents_seed${s}.pt --log_path run_log_5mediumagents_seed${s}.pt; done
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
tmux
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5easyagents_seed${s}.pt --log_path run_log_5easyagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 4 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5mediumagents_seed${s}.pt --log_path run_log_5mediumagents_seed${s}.pt; done
tmux
tmux
tmux
tmux
tmux
tmux attach -t 4
tmux attach -t 4
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents-seed1.pt run_log_20hardagents-seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents-seed1.pt run_log_20hardagents-seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux ls
tmux kill -8
tmux kill-session -t 8
tmux kill-session -t 9
tmux kill-session -t 10
tmux kill-session -t 11
tmux kill-session -t 12
tmux ls
tmux
tmux ls
tmux ls
tmux attach -t 4
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.0001 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5easyagents_seed${s}.pt --log_path run_log_5easyagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.0001 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5mediumagents_seed${s}.pt --log_path run_log_5mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5easyagents_seed${s}.pt --log_path run_log_5easyagents_seed${s}.pt; done
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt run_log_10agents_seed2.pt run_log_20hardagents_seed2.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed3.pt run_log_10agents_seed3.pt run_log_20hardagents_seed3.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt run_log_10agents_seed2.pt run_log_20hardagents_seed2.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux -ls
tmux ls
tmux kill-session -t 13
tmux kill-session -t 14
tmux kill-session -t 15
tmux kill-session -t 16
tmux kill-session -t 17
tmux kill-session -t 18
tmux -ls
tmux ls
tmux 
tmux
tmux ls
tmux kill-session -t 19
tmux kill-session -t 20
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
tmux
tmux
tmux
tmux
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 256 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20hardagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux attach -t 4
tmux -ls
tmux ls
tmux kill-session -t 23
tmux kill-session -t 24
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5mediumagents_seed${s}.pt --log_path run_log_5mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
tmux
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux ls
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed2.pt --agent_counts 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt --agent_counts 5 --smooth 100
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
git commit -m "first eval try"
git add .
git commit -m "first eval try"
git add .
git comit -m "eval does not change training"
git commit -m "eval does not change training"
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux 'ls
tmux -ls
tmux ls
tmux kill-session -t 21
tmux kill-session -t 25
tmux kill-session -t 26
tmux kill-session -t 27
tmux
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_5easyagents_seed2.pt run_log_5easyagents_seed3.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt run_log_10agents_seed2.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt run_log_10agents_seed2.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed3.pt run_log_10agents_seed2.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux -ls
tmux ls
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_10agents_seed1.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed2.pt run_log_10agents_seed2.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed3.pt run_log_10agents_seed3.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed4.pt run_log_10agents_seed3.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed5.pt run_log_10agents_seed3.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed5.pt run_log_10agents_seed3.pt run_log_20mediumagents_seed1.pt --agent_counts 5 10 20 --smooth 100
tmux ls
tmux attach -t 0
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 500 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-15.pt --log_path run_log_5agents-15.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --save model_5agents-15.pt --log_path run_log_5agents-15.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 18 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --save model_20agents-15.pt --log_path run_log_20agents-15.pt
python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 3 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 12 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_10agents-15.pt --log_path run_log_10agents-15.pt
python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 18 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_20agents-15medium.pt --log_path run_log_20agents-15medium.pt
python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --save model_5agents-15medium.pt --log_path run_log_5agents-15medium.pt
tmux
git add .
git restore main.py
git checkout 52c49399e8920c5e9e45e49721c967d95156a6b4 -- main.py
git show 52c49399e8920c5e9e45e49721c967d95156a6b4:main.py
git log
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_10agents_Atnseed${s}.pt --log_path run_log_10agents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_10agents_Atnseed${s}.pt --log_path run_log_10agents_Atnseed${s}.pt; done
tmux
tmux
tmux
tmux ls
tmux kill-session -t 32
tmux
tmux ls
tmux kill-session -t 0
tmux kill-session -t 1
tmux kill-session -t 2
tmux kill-session -t 3
tmux kill-session -t 4
tmux kill-session -t 5
tmux kill-session -t 6
tmux kill-session -t 7
tmux ls
tmux
tmux
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20hardagents_seed${s}.pt --log_path run_log_20hardagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4--save model_5easyagents_Atnseed${s}.pt --log_path run_log_5easyagents_Atnseed${s}.pt; done
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-NoAtn3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard2.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-NoAtn3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard2.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-NoAtn3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-NoAtn3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-NoAtn3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard2.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3medium.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard2.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-pasthope3-2.pt run_log_20agents-deletecrashedcars3hard2.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 21 --max_steps 1000 --ic3net --vision 2 --difficulty hard --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4--save model_20hardagents_Atnseed${s}.pt --log_path run_log_20hardagents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_20mediumagents_Atnseed${s}.pt --log_path run_log_20mediumagents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_10agents_Atnseed${s}.pt --log_path run_log_10agents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 10 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 2000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 14 --max_steps 500 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_10agents_seed${s}.pt --log_path run_log_10agents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_5mediumagents_Atnseed${s}.pt --log_path run_log_5mediumagents_Atnseed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 3000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5mediumagents_seed${s}.pt --log_path run_log_5mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty easy --crash_penalty -15 --terminal_reward 15 --seed $s --save model_5easyagents_seed${s}.pt --log_path run_log_5easyagents_seed${s}.pt; done
git remote -v
git remote remove origin
git remote add origin https://github.com/Aleksandra39363/MastersProject.git
git remote -v
git push -u origin main
git branch
git push -u origin master
git rm -r --cached .vscode-server
git commit -m "Remove vscode server files"
git add visualize_training.py
git checkout HEAD~1 -- visualize_training.py
git add visualize_training.py
git commit -m "Restore visualization"
git push -f origin master
rm -rf .git
git init
git add .
git commit -m "Initial clean commit"
git branch -M master
git remote add origin https://github.com/Aleksandra39363/MastersProject.git
git push -u origin master
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
conda env
conda activate thesis
which conda
module load anaconda
ls
source thesis/bin/activate
ls thesis/bin
which conda
conda activate multiagent
source multiagent/bin/activate
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
conda activate thesis
conda deactivate
conda deactivate
conda activate thesis
tmux
conda activate thesis
for s in 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 5 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 1 --num_epochs 1000 --epoch_size 20 --batch_size 150 --hid_size 128 --detach_gap 10 --lrate 0.00005 --dim 10 --max_steps 300 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --use_agent_attn --attn_heads 4 --save model_5mediumagents_Atnseed${s}.pt --log_path run_log_5mediumagents_Atnseed${s}.pt; done
conda activate thesis
tmux
conda activate thesis
tmux
conda create -n project python=3.12 -y
conda activate project
conda create -n project python=3.12 -y
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n thesis python=3.12 -y
conda activate thesis
pip install -r requirements.txt
conda install -c conda-forge --file requirements.txt
pip install -r requirements.txt
pip install --upgrade pip setuptools wheel
pip install visdom
pip install -r requirements.txt
pip install -r requirements.txt
pip install -r requirements.txt
python3 visualize_training.py --compare-scaling --log_files run_log_2agents-deletecrashedcars3.pt run_log_5agents-pasthope3easy2.pt run_log_10agents-deletecrashedcars3-2.pt run_log_20agents-deletecrashedcars3hard.pt --agent_counts 2 5 10 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_5easyagents_seed2.pt run_log_5easyagents_seed3.pt run_log_4easyagents_seed5.pt run_log_5easyagents_seed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_5mediumagents_seed2.pt run_log_5mediumagents_seed3.pt run_log_4mediumagents_seed5.pt run_log_5mediumagents_seed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5easyagents_seed1.pt run_log_5easyagents_seed2.pt run_log_5easyagents_seed3.pt run_log_5easyagents_seed4.pt run_log_5easyagents_seed5.pt --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_5mediumagents_seed2.pt run_log_5mediumagents_seed3.pt run_log_5mediumagents_seed4.p run_log_5mediumagents_seed5.p  --agent_counts 10 10 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_5mediumagents_seed2.pt run_log_5mediumagents_seed3.pt run_log_5mediumagents_seed4.p run_log_5mediumagents_seed5.p  --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_5mediumagents_seed1.pt run_log_5mediumagents_seed2.pt run_log_5mediumagents_seed3.pt run_log_5mediumagents_seed4.pt run_log_5mediumagents_seed5.pt  --agent_counts 5 5 5 5 5 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_seed1.pt run_log_10agents_seed2.pt run_log_10agents_seed3.pt  --agent_counts 10 10 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_10agents_Atnseed1.pt --agent_counts 10 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_Atnseed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt run_log_20mediumagents_seed2.pt --agent_counts 20 20  --smooth 100
tmux
conda activate thesis
tmux
conda activate thesis
python3 visualize_training.py --compare-scaling --log_files run_log_20hardagents_seed2.pt run_log_20hardagents_seed1.pt --agent_counts 20 20 --smooth 100
pip install visdom
pip install -e ./ic3net_envs
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
for s in 1 2 3 4 5; do python3 main.py --env_name traffic_junction --nagents 20 --add_rate_min 1.0 --add_rate_max 1.0 --nprocesses 8 --num_epochs 2000 --epoch_size 10 --batch_size 500 --hid_size 256 --comm_passes 3 --entr 0.001 --detach_gap 10 --lrate 0.0005 --dim 20 --max_steps 1000 --ic3net --vision 2 --difficulty medium --crash_penalty -15 --terminal_reward 15 --seed $s --save model_20mediumagents_seed${s}.pt --log_path run_log_20mediumagents_seed${s}.pt; done
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1-sporo.pt --agent_counts 20 --smooth 100
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
python3 visualize_training.py --compare-scaling --log_files run_log_20mediumagents_seed1.pt --agent_counts 20
