Multi-Head Attention for Scalable Multi-Agent Coordination
========================================================

Abstract
--------
We integrate a multi-head self-attention module into an IC3Net-style multi-agent policy to improve coordination in the Traffic Junction task. The attention block operates across agents' hidden states and replaces or complements the original hard communication gate. Experiments show improved coordination up to ~25 agents compared to the baseline hard-gating CommNet, but diminishing returns and increased fragility beyond ~30 agents. We document the model changes, underlying reasons for improvement, empirical behaviour, computational trade-offs, and practical recommendations for scaling.

1. Introduction
----------------
Multi-agent coordination in partially-observed, dynamic environments is a core challenge for decentralized RL. The Traffic Junction benchmark requires cars to traverse shared intersections without crashing; success is a strict team-level metric (all cars complete without crashes). The baseline IC3Net/CommNet approach uses a learned per-agent encoder and a communication mechanism that either averages messages or uses a binary talk/silent action. We augment this with a learned multi-head self-attention mixer over agent hidden states and evaluate the consequences.

2. Background and Motivation
----------------------------
- Baseline: `CommNetMLP` implements per-agent encoding and communication via averaged/summed messages and an optional hard gate (`comm_action`). See [main.py](main.py#L28) and the baseline code.
- Attention idea: multi-head attention allows per-agent queries to attend to all agent keys/values, enabling selective, learned relational mixing. The attention implementation used here is in [comm_attention.py](comm_attention.py#L47).

Motivation: attention can (a) represent heterogeneous, context-dependent interactions, (b) let different heads specialize on different relation types, and (c) provide dense gradients for credit assignment — all desirable in coordination tasks.

3. Model: where attention is added
---------------------------------
Architecture (high level):

1. Shared encoder: a linear projection maps each agent's observation to a `hidden_size` vector.
2. Agent mixer: reshape to `(batch, N_agents, hidden_size)` and apply `nn.MultiheadAttention` (the core call is in [comm_attention.py](comm_attention.py#L115)).
3. Residual gating & normalization: attended output is gated and combined with a residual connection (see [comm_attention.py](comm_attention.py#L123)).
4. Optional recurrence: a GRU/LSTM/MAMBA block processes per-agent state in time.
5. Action heads and value head produce outputs for policy and baseline.

Concrete code references:

- The attention layer is constructed at [comm_attention.py](comm_attention.py#L47).
- The multi-pass communication loop (if `--comm_passes > 1`) applies attention repeatedly at [comm_attention.py](comm_attention.py#L110-L122).
- Residual gating (soft continuous gate) is implemented at [comm_attention.py](comm_attention.py#L123).

4. Theoretical advantages
-------------------------
- Relational expressivity: Attention learns pairwise importance weights W_{ij}(t) so agent i's update is a weighted combination of other agents' representations.
- Multiple subspaces: `H` heads provide different projection subspaces; each head can attend to a different interaction pattern (blocker, route-conflict, flow leader).
- Dense credit assignment: continuous attention weights provide nonzero gradients even when binary comm actions would be saturated.

5. Computational and optimization trade-offs
------------------------------------------
- Complexity: full self-attention scales with O(N_agents^2 × head_dim). Going from 25→30 agents increases pairwise terms by 44% (625→900).
- Memory: attention stores per-head weights; multi-head and larger `hidden_size` increase peak memory usage.
- Optimization sensitivity: more parameters and denser interactions increase sensitivity to LR, entropy, gradient clipping, and curriculum.

6. Experimental Protocol
------------------------
We follow the project training setup (see `train_20agents_fixed.sh`) with the following adaptions for attention experiments:

- Environment: Traffic Junction (see `ic3net_envs/ic3net_envs/traffic_junction_env.py`).
- Baseline flags: `--ic3net` (enables `commnet` + hard gate); Base uses `CommNetMLP` by default ([main.py](main.py#L28)).
- Flash/Attention runs use the `--flash` flag to activate attention ([main.py](main.py#L172)).
- Key hyperparameters used in your runs: `hid_size=128`, `comm_passes=1`, `entr` small positive in stage 1 then 0 in stage 2 (see `train_20agents_fixed.sh`).

Suggested evaluation metrics (already logged in trainer):

- Episode return (sum of per-agent rewards), per-agent completion rate, total crashes, and strict team success (all complete & no crashes). See logging in `main.py` and `trainer.py`.

7. Empirical findings (summary)
-------------------------------
- The attention variant consistently outperforms the base hard-gate CommNet up to about N≈25 agents: higher per-agent completion, fewer collisions, and improved episode reward.
- Around N≈30 the attention run becomes fragile: instability, higher variance across seeds, and degraded team success despite modest per-agent gains.

Interpretation: attention helps because it enables selective relational reasoning; the fragility at larger N comes from increased interaction noise, optimization difficulty, and the harsh team-level success metric.

8. Practical recommendations
---------------------------
To make attention work reliably at larger scales, apply a combination of improvements:

1. Increase representational capacity: raise `hid_size` so each attention head has more dimensions (e.g., 256+). See discussion in `IC3NET_SCALING_ANALYSIS.md`.
2. Use hierarchical attention / locality bias: restrict attention to spatially or topologically local neighborhoods, or implement a hierarchical team-leader attention (pattern shown by `HierarchicalAttentionCommNet` in `comm_attention.py`).
3. Stronger curriculum and reward shaping: ramp agent spawn rate slowly and add intermediate progress rewards to densify learning signal.
4. Optimization stability: lower LR, add entropy regularization early, clip gradients, and increase batch size.
5. Ablation-first approach: stabilize reward & optimization before swapping architectures; then compare Base vs Attention vs Hierarchical attention under stable training.

9. Limitations
--------------
- Attention is not a silver bullet: pairwise complexity and optimization fragility limit raw scaling.
- Strict team success amplifies sparsity — per-agent improvements may not translate into team success without focused training objectives.

10. Conclusion
--------------
Multi-head self-attention significantly improves relational modeling and coordination in the Traffic Junction environment at moderate team sizes (≤25). The model gains expressivity and interpretability, but confronts complexity and optimization issues as N grows. Practical scaling requires combining attention with architecture-level biases (hierarchical/local attention) and stronger training design (curriculum, reward shaping, and optimizer tuning).

References and code pointers
---------------------------
- Implementation: [comm_attention.py](comm_attention.py#L47)
- Baseline selection: [main.py](main.py#L28) and the `--flash` flag handling at [main.py](main.py#L172-L199)
- Environment reward & termination: [ic3net_envs/ic3net_envs/traffic_junction_env.py](ic3net_envs/ic3net_envs/traffic_junction_env.py#L713)

Appendix: Quick run suggestions
------------------------------
To reproduce attention runs (example):

```bash
# Stage 1: small entropy for exploration
python3 main.py --env_name traffic_junction --nagents 25 --flash --attn_heads 4 \
  --hid_size 128 --lrate 5e-5 --num_epochs 800 --entr 5e-5 --seed 1 \
  --save model_25agents_Flash_s1_stage1.pt --log_path run_log_25agents_Flash_s1_stage1.pt

# Stage 2: fine-tune with entr=0
python3 main.py --env_name traffic_junction --nagents 25 --flash --attn_heads 4 \
  --hid_size 128 --lrate 5e-5 --num_epochs 1200 --entr 0.0 --seed 1 \
  --load model_25agents_Flash_s1_stage1.pt --save model_25agents_Flash_s1_stage2.pt \
  --log_path run_log_25agents_Flash_s1_stage2.pt
```

If you want, I can convert this document into a LaTeX-ready section or expand the Methods/Results with tables and plots from your logs.
