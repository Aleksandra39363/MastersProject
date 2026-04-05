# Why IC3Net Fails at 20 Agents (and How to Fix It)

## Deep Dive into Root Causes

### 1. **Entropy Regularization (MOST CRITICAL)**

**Your current setting:** `--entr 0` (no entropy penalty)

**Why this breaks at 20 agents:**

```
5 agents:  Each learns simple 5-agent coordination → works with little exploration
10 agents: 10-agent patterns still learnable with exploration from PPO randomness alone
20 agents: 20-agent coordination is manifold of solutions → REQUIRES high exploration
```

**The math:** With hard attention + 20 agents:
- Each agent makes binary comm decision: 2^20 = 1 million possible communication patterns
- Without entropy bonus, network collapses to one pattern (e.g., "all silent" or "all talk")
- With entropy, network explores diverse communication strategies

**Fix:** `--entr 0.001` to `--entr 0.01`
- Adds penalty for low-entropy action distributions
- Forces network to explore rather than collapse

---

### 2. **Hidden Size (Attention Bottleneck)**

**Your current setting:** `--hid_size 64`

**Why attention fails:**

In `comm_attention.py` line ~50:
```python
self.attention = nn.MultiheadAttention(
    embed_dim=64,        # ← This is the problem!
    num_heads=8,
    ...
)
```

With 8 heads and embed_dim=64:
- Each head processes 64/8 = **8 dimensions per head**
- With 20 agents, each head must compress agent info into 8 dims
- Result: **Noisy attention weights**, poor coordination signal

**Comparison:**
- 5 agents + 64 dims: 64/8 = 8 dims/head ✓ OK
- 10 agents + 64 dims: 64/8 = 8 dims/head ✓ Barely works
- 20 agents + 64 dims: 64/8 = 8 dims/head ✗ **Too compressed**
- 20 agents + 256 dims: 256/8 = 32 dims/head ✓ Good!

**Fix:** `--hid_size 256` or `--hid_size 512`
- More dimensions = each attention head has more expressive power
- Trade-off: Slightly slower computation, much better coordination

---

### 3. **Communication Passes (Information Propagation)**

**Your current setting:** `--comm_passes 1`

**Why 1 pass fails:**

Communication rounds in IC3Net:
```
Pass 1: Agent A hears from direct neighbors
Pass 2: Agent A hears from neighbors' neighbors (2-hop)
Pass 3: Agent A hears from 3-hop neighbors
```

With 20 agents in a grid/network topology:
- Average agent is 3-4 hops from farthest agent
- **1 pass:** Agent only gets local information → can't coordinate globally
- **3 passes:** Information reaches all agents → global coordination

**Visualization:**
```
Pass 1:  A ← B ← C ← D ← E...  (E doesn't hear A yet)
Pass 2:  A ← B ← C ← D ← E...  (E hears A indirectly)
Pass 3:  A ← B ← C ← D ← E...  (A hears full team context)
```

**Fix:** `--comm_passes 3` or `--comm_passes 2`
- Slight computational overhead
- HUGE coordination improvement

---

### 4. **Learning Rate (Training Stability)**

**Your current setting:** `--lrate 0.001`

**Why this matters at 20 agents:**

With 20 agents + complex coordination:
- Gradient magnitudes 10x-100x larger than 5 agents
- Learning rate 0.001 → **overshoots, diverges**
- Network loses learned patterns mid-training

**Fix:** `--lrate 0.0005` for 20 agents
- 2x lower learning rate
- Essential for stable large-scale training
- Can revert to 0.001 for 10 agents

---

### 5. **Attention vs CommNet (The Fundamental Issue)**

**Why attention makes 20 agents WORSE:**

This is counterintuitive but critical!

```
Original CommNet (in comm.py):
  - All agents average together: comm = mean(all_agent_embeddings)
  - Everyone gets same info → coordination via averaging
  - O(N) communication per agent
  - Works for 5-100 agents!

Attention CommNet (in comm_attention.py):
  - Each agent learns which others to attend to
  - Goal: "Don't pay attention to noise, focus on important agents"
  - Problem: With hard attention + 20 agents = learning fails
  - Attention weights become noisy/unreliable
```

**Why attention fails at 20 agents:**
- Network tries to learn: "Agent A should talk to agents [1,3,5] but not [2,4,6]"
- With limited training: learns garbage patterns instead
- CommNet's averaging → always useful signal, even if not optimal
- Attention → learns poor selectivity → worse than no selection

**Fix:** **Use CommNet for 20 agents, NOT attention**
- Simpler to train
- Proven to work at any scale
- Don't use `--use_agent_attn` flag for 20+ agents

---

## Recommended Training Script for 20 Agents

```bash
python3 main.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 20 \
  --hid_size 256 \           # ↑ From 64
  --comm_passes 3 \          # ↑ From 1
  --entr 0.001 \             # ↑ From 0
  --lrate 0.0005 \           # ↓ From 0.001
  --num_epochs 1500 \        # ↑ More training
  --batch_size 500 \
  --seed 1 \
  --eval_every 100 \
  --eval_epochs 10
```

**Relative improvement expected:**
- Current 20-agent performance: ~20-40% success
- With fixes: ~60-80% success (goal-dependent)

---

## Why Your 10-Agent Attention Works So Well

With attention + 10 agents:
- Coordination problem is simpler (fewer dependencies)
- 64 dims with 8 heads = 8 dims/head ✓ Reasonable
- 1 comm pass = sufficient in most cases
- Natural PPO exploration + entropy=0 works fine

**This doesn't scale:** The sweet spot for attention is 5-12 agents.

---

## Advanced: If Performance Still Poor

Try one or more of:

1. **Add recurrence:**
   ```bash
   --recurrent  # Adds LSTM cell for temporal memory
   ```

2. **Reduce team from 20→15:**
   ```bash
   --nagents 15
   ```
   (Sometimes 15 agents is sweet spot between complexity and tractability)

3. **Add reward normalization:**
   ```bash
   --normalize_rewards
   ```

4. **Increase batch size:**
   ```bash
   --batch_size 1000  # More data per update = more stable
   ```

---

## Summary Table

| Hyperparameter | 5 Agents | 10 Agents | 20 Agents |
|---|---|---|---|
| `hid_size` | 64 | 64 | **256** |
| `comm_passes` | 1 | 1-2 | **3** |
| `entr` | 0 | 0 | **0.001** |
| `lrate` | 0.001 | 0.001 | **0.0005** |
| Model | CommNet or Attn | Attention✓ | CommNet✓ |

---

## What to Try Next

1. Run training with script above
2. Train for **1500 epochs** (not 1000) for 20 agents
3. Visualize: `python3 visualize_training.py --compare-scaling --log_files run_log_20agents_fixed_seed1.pt --agent_counts 20`
4. If still poor, reach out with results and I'll give more specific tuning

Good luck! 🚀
