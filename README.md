# TorchCode

**Crack the PyTorch interview.** Practice implementing operators and architectures from scratch — the exact skills top ML teams test for.

> Like LeetCode, but for tensors. Self-hosted. Jupyter-based. Instant feedback.

## Why TorchCode?

Top companies (Meta, Google DeepMind, OpenAI, etc.) expect ML engineers to implement core operations **from memory on a whiteboard**. Reading papers isn't enough — you need to write `softmax`, `LayerNorm`, `MultiHeadAttention`, and full Transformer blocks cold.

TorchCode gives you a **structured practice environment** with:

- **13 curated problems** covering the most frequently asked PyTorch interview topics
- **Automated judge** with correctness checks, gradient verification, and timing
- **Instant feedback** — colored pass/fail per test case, just like competitive programming
- **Hints when stuck** — nudges without full spoilers
- **Reference solutions** — study optimal implementations after your attempt
- **Progress tracking** — see what you've solved, your best times, and attempt counts

No cloud. No signup. No GPU needed. Just `make run`.

## Quick Start

### Option 1 — Pull the pre-built image (fastest)

```bash
docker run -p 8888:8888 ghcr.io/duoan/torchcode:latest
```

### Option 2 — Build locally

```bash
make run
```

Open **http://localhost:8888** — that's it. Works with both Docker and Podman (auto-detected).

## Problem Set

### Fundamentals — "Implement X from scratch"

The bread and butter of ML coding interviews. You'll be asked to write these without `torch.nn`.

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|---|---------|----------------------|------------|--------------|
| 1 | ReLU | `relu(x)` | Easy | Activation functions, element-wise ops |
| 2 | Softmax | `my_softmax(x, dim)` | Easy | Numerical stability, exp/log tricks |
| 3 | Linear Layer | `SimpleLinear` (nn.Module) | Medium | `y = xW^T + b`, Kaiming init, `nn.Parameter` |
| 4 | LayerNorm | `my_layer_norm(x, gamma, beta)` | Medium | Normalization, running stats, affine transform |
| 7 | BatchNorm | `my_batch_norm(x, gamma, beta)` | Medium | Batch vs layer statistics, train/eval behavior |
| 8 | RMSNorm | `rms_norm(x, weight)` | Medium | LLaMA-style norm, simpler than LayerNorm |

### Attention Mechanisms — The heart of modern ML interviews

If you're interviewing for any role touching LLMs or Transformers, expect at least one of these.

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|---|---------|----------------------|------------|--------------|
| 5 | Scaled Dot-Product Attention | `scaled_dot_product_attention(Q, K, V)` | Hard | `softmax(QK^T/√d_k)V`, the foundation of everything |
| 6 | Multi-Head Attention | `MultiHeadAttention` (nn.Module) | Hard | Parallel heads, split/concat, projection matrices |
| 9 | Causal Self-Attention | `causal_attention(Q, K, V)` | Hard | Autoregressive masking with `-inf`, GPT-style |
| 10 | Grouped Query Attention | `GroupQueryAttention` (nn.Module) | Hard | GQA (LLaMA 2), KV sharing across heads |
| 11 | Sliding Window Attention | `sliding_window_attention(Q, K, V, w)` | Hard | Mistral-style local attention, O(n·w) complexity |
| 12 | Linear Attention | `linear_attention(Q, K, V)` | Hard | Kernel trick, `φ(Q)(φ(K)^TV)`, O(n·d²) |

### Full Architecture — Put it all together

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|---|---------|----------------------|------------|--------------|
| 13 | GPT-2 Block | `GPT2Block` (nn.Module) | Hard | Pre-norm, causal MHA + MLP (4x, GELU), residual connections |

## How It Works

Each problem has **two** notebooks:

| File | Purpose |
|------|---------|
| `01_relu.ipynb` | Blank template — write your code here |
| `01_relu_solution.ipynb` | Reference solution — check when stuck |

### Workflow

```
1. Open a blank notebook           →  Read the problem description
2. Implement your solution         →  Use only basic PyTorch ops
3. Debug freely                    →  print(x.shape), check gradients, etc.
4. Run the judge cell              →  check("relu")
5. See instant colored feedback    →  ✅ pass / ❌ fail per test case
6. Stuck? Get a nudge              →  hint("relu")
7. Review the reference solution   →  01_relu_solution.ipynb
```

### In-Notebook API

```python
from torch_judge import check, hint, status

check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint without full spoiler
status()                    # Progress dashboard — solved / attempted / todo
```

## Suggested Study Plan

**Week 1 — Foundations** (warm-up, 1–2 hours)
- Day 1: ReLU, Softmax
- Day 2: Linear Layer
- Day 3: LayerNorm, BatchNorm, RMSNorm

**Week 2 — Attention Deep Dive** (interview-critical, 3–4 hours)
- Day 1: Scaled Dot-Product Attention
- Day 2: Multi-Head Attention
- Day 3: Causal Self-Attention
- Day 4: GQA, Sliding Window, Linear Attention

**Week 3 — Integration** (1–2 hours)
- Day 1: GPT-2 Block (combines everything)
- Day 2: Speed run — re-implement all from scratch, timed

## Architecture

```
┌──────────────────────────────────────────┐
│           Docker / Podman Container      │
│                                          │
│  JupyterLab (:8888)                     │
│    ├── templates/  (reset on each run)  │
│    ├── solutions/  (reference impl)     │
│    ├── torch_judge/ (auto-grading)      │
│    └── PyTorch (CPU), NumPy             │
│                                          │
│  Judge checks:                           │
│    ✓ Output correctness (allclose)      │
│    ✓ Gradient flow (autograd)           │
│    ✓ Shape consistency                  │
│    ✓ Edge cases & numerical stability   │
└──────────────────────────────────────────┘
```

Single container. Single port. No database. No frontend framework. No GPU.

## Commands

```bash
make run    # Build & start (http://localhost:8888)
make stop   # Stop the container
make clean  # Stop + remove volumes + reset all progress
```

## Adding Your Own Problems

TorchCode uses auto-discovery — just drop a new file in `torch_judge/tasks/`:

```python
TASK = {
    "id": "my_task",
    "title": "My Custom Problem",
    "difficulty": "medium",
    "function_name": "my_function",
    "hint": "Think about broadcasting...",
    "tests": [ ... ],
}
```

No registration needed. The judge picks it up automatically.

## FAQ

**Q: Do I need a GPU?**
A: No. Everything runs on CPU. The problems test correctness and understanding, not throughput.

**Q: Can I keep my solutions between runs?**
A: Blank templates reset on every `make run` so you practice from scratch. Save your work under a different filename if you want to keep it.

**Q: How are solutions graded?**
A: The judge runs your function against multiple test cases using `torch.allclose` for numerical correctness, verifies gradients flow properly via autograd, and checks edge cases specific to each operation.

## License

MIT
