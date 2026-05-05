# Multi-Head Attention in the Traffic Junction Model

## Overview

The attention-based variant is intended to replace the simpler IC3Net communication bottleneck with learned agent-to-agent mixing. In practice, the model still starts from the same per-agent observation encoder, but it inserts a multi-head self-attention block over the agent dimension before the action heads.

This makes the policy more expressive because each agent can condition its decision on the hidden states of the other agents instead of relying only on a gated binary communication choice.

## Where It Is Added

The base entry point in [main.py](main.py#L28) imports `CommNetMLP` from `comm.py`. The attention path is activated by the `--flash` flag in [main.py](main.py#L172), which keeps communication enabled and switches the model toward the attention-based agent mixer. The attention implementation itself lives in [comm_attention.py](comm_attention.py#L47).

At a high level, the model does this:

1. Encode each agent’s observation with a shared linear layer.
2. Reshape the batch into `(batch, nagents, hidden_size)`.
3. Apply `nn.MultiheadAttention` across the agent axis.
4. Pass the attended agent states through a residual gate and normalization.
5. Optionally run a recurrent layer.
6. Produce one action head per action dimension.

The core attention call is in [comm_attention.py](comm_attention.py#L115), and the residual gating happens immediately after in [comm_attention.py](comm_attention.py#L123).

## What Multi-Head Attention Changes

Multi-head attention replaces a single fixed communication rule with several learned communication subspaces.

Instead of asking only, “should this agent talk or stay silent?”, the model learns:

- which other agents matter right now,
- which interaction pattern is relevant for this head,
- and how to combine multiple interaction patterns into one state update.

This is important in traffic junction because the relevant neighbors are not always the same. Sometimes the key signal is a car directly ahead; sometimes it is a car in a conflicting lane; sometimes it is a chain of cars whose movement determines whether the junction clears.

## Why It Can Improve Performance

### 1. Better relational modeling

Traffic junction is fundamentally a coordination problem. The action for one car depends on the state of other cars, not just its own local observation. Attention gives the model a direct way to represent those dependencies.

### 2. Learnable selective focus

The model does not have to average over every agent equally. It can assign larger weights to the agents that are most relevant for the current decision. That is useful when only a subset of cars are in conflict.

### 3. Multiple communication patterns at once

Different heads can specialize. One head may track immediate blockers, another may track route conflicts, and another may capture broader traffic flow. That is more flexible than a single communication gate.

### 4. Better credit assignment for coordination

Because the attention weights are learned, gradients can push the model toward interactions that actually help reward, instead of forcing it to learn one global compression for all agent relations.

### 5. More stable than discrete communication choices

Compared with a hard talk/silent switch, soft attention provides a dense signal. That usually makes training easier because the model gets continuous feedback about which agents influenced the decision.

## Why It Struggles As Agent Count Grows

The fact that it works well around 25 agents but starts to struggle around 30 is not surprising.

### 1. Attention cost grows quickly

Full self-attention mixes every agent with every other agent. That means the interaction space grows roughly with $N^2$ as the number of agents increases. Going from 25 to 30 agents may look small, but the number of pairwise interactions rises from 625 to 900, which is a 44% increase.

### 2. More agents means more irrelevant interactions

As the team gets larger, the model must learn to suppress a growing amount of noise. That makes optimization harder, because the useful signal is buried under many more possible pairwise relations.

### 3. Coordination becomes harder than local reaction

With 25 agents, the policy may still discover a useful set of traffic rules. At 30 agents, the same grid and reward structure can create more congestion, more simultaneous conflicts, and more brittle timing dependencies. The policy then needs not just attention, but very strong temporal and structural bias.

### 4. Reward signal becomes sparse at the team level

The environment uses strict success criteria: all cars must complete without crashes. That metric gets harsher as the number of agents rises. Even if average behavior improves, the probability that all 30 agents coordinate perfectly is much lower than for 25.

### 5. Optimization sensitivity increases

With more agents, training becomes more sensitive to learning rate, entropy regularization, gradient clipping, and curriculum design. A setup that is stable at 20 or 25 agents can become brittle at 30.

## Consequences For The Model

### Positive consequences

- Better long-range coordination than the base hard-communication model.
- More interpretable interaction structure through attention weights.
- Better ability to learn non-uniform importance across agents.
- Often improved performance at moderate agent counts.

### Negative consequences

- Higher compute and memory cost than a simple communication gate.
- More parameters and more optimization sensitivity.
- More fragile when the number of agents grows.
- Can overfit to attention patterns that work at one scale but fail at a larger scale.

## Practical Interpretation Of Your Results

Your observation is consistent with a model that is genuinely better than the base policy at moderate scale, but still limited by full agent-to-agent attention. If the model behaves well at 25 agents and then starts struggling at 30, that usually means the architecture is near its coordination capacity rather than completely broken.

In other words:

- The attention mechanism is helping.
- The improvement is real.
- But the benefit is not unbounded.
- Past a certain agent count, the coordination problem outgrows the model’s capacity and the training setup.

## Conclusion

Multi-head attention is a strong upgrade over binary communication because it lets agents learn which other agents matter and how different interaction patterns should be combined. In this codebase, it is added as a self-attention block over the agent hidden states before the action heads. That generally improves coordination, especially at moderate scale.

However, attention is not a free scaling solution. As the number of agents increases, the pairwise interaction space, the amount of irrelevant traffic, and the difficulty of strict team success all grow quickly. That is why a configuration can look solid at 25 agents and still start to degrade at 30.

The main practical lesson is that attention improves the policy, but it does not eliminate the need for a good curriculum, reward shaping, and careful optimization.