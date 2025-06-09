# CIv6-SBD Solution Proposal: Geometric-Topological Structural Break Detection

## 🚧 Updated Implementation Plan (Aligned with ADIA Challenge + Topological Reasoning Engine)

### 🧠 Overview

This solution expands the original proposal by integrating concrete code from a working topological reasoning notebook. It aligns directly with the ADIA Lab Structural Break Detection challenge and operationalizes the CIv6 hypothesis through latent geometry and attention topology.

---

## 1. Preprocessing and ECA Encoding

### Input

* `X_train`: Pandas DataFrame, indexed by `id`, with `value` and `period` columns.
* `y_train`: Boolean labels indicating structural breaks.
* `X_test`: List of test DataFrames.

### Transformation Pipeline

1. **Symbolic Encoding**

   * Convert each time series segment (pre/post boundary) into symbolic binary strings.
   * Use delta-sign encoding or permutation-based symbolic embedding.

2. **ECA Evolution**

   * Apply Elementary Cellular Automata (e.g. Rule 110) to each binary sequence.
   * Run for 32–64 time steps.
   * Flatten 2D evolution into token-like input strings.

```python
# Example: run_eca function
RULE_MAPS = {
    110: {f"{i:03b}": b for i, b in enumerate(f"{110:08b}"[::-1])}
}

def run_eca(initial_bits, rule=110, steps=32):
    rule_map = RULE_MAPS[rule]
    current = ''.join(initial_bits)
    history = [current]
    for _ in range(steps):
        padded = '0' + current + '0'
        next_state = ''.join(rule_map[padded[i:i+3]] for i in range(len(current)))
        history.append(next_state)
        current = next_state
    return history
```

---

## 2. Model: Topological Attention Analyzer

### Model Architecture

* Use a Transformer model capable of outputting `attentions` and hidden `embeddings`.

  * Pretrained options: `GPT-2`, `Qwen2.5`, or `TransformerECA` (if symbolically trained).

### Core Modules

1. **Semantic Loop Monitor**

```python
def extract_cycles_and_log_wilson(head_idx, attn_matrix, tokens, threshold=0.05):
    import networkx as nx
    import numpy as np
    A = attn_matrix[head_idx].cpu().numpy()
    idx = {t: i for i, t in enumerate(tokens)}
    G = nx.DiGraph()
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if A[i, j] > threshold:
                G.add_edge(tokens[i], tokens[j], weight=A[i, j])
    cycles = [c for c in nx.simple_cycles(G) if 3 <= len(c) <= 6]

    def log_wilson_loop(cycle):
        return sum(np.log(A[idx[s], idx[t]] + 1e-12) for s, t in zip(cycle, cycle[1:] + cycle[:1]))

    return [(cycle, log_wilson_loop(cycle)) for cycle in cycles]
```

2. **Holonomy Analyzer**

```python
def compute_holonomy_spectrum(cycle, Q, K, token_idx):
    import torch, numpy as np
    H = torch.eye(Q.shape[-1])
    for s, t in zip(cycle, cycle[1:] + cycle[:1]):
        i, j = token_idx[s], token_idx[t]
        qi = Q[i].unsqueeze(1)
        kj = K[j].unsqueeze(0)
        transport = qi @ kj
        H = transport @ H
    eigvals = torch.linalg.eigvals(H).cpu().numpy()
    return eigvals
```

3. **Topological Divergence Detector**

   * Compare:

     * Loop energy stats (mean, max, count)
     * Eigenvalue spectrum: dispersion, real-imag range
     * Use statistical distance (e.g., cosine/Mahalanobis) to assess change

4. **Entropy Divergence Tracker**

   * Use attention matrices to compute entropy per token:

```python
def compute_attention_entropy(attn_matrix):
    import scipy.stats
    entropy_per_head = []
    for head_attn in attn_matrix:
        probs = head_attn / head_attn.sum(axis=-1, keepdims=True)
        entropy = scipy.stats.entropy(probs, axis=-1)
        entropy_per_head.append(entropy.mean())
    return entropy_per_head
```

---

## 3. Detection Logic

### Break Score Computation

* Define break score as:

  * Drop in mean loop energy
  * Spectral divergence (eigenvalue distance)
  * Entropy jump or FIM curvature collapse
* Can be a trained classifier or a rule-based scoring function.

### Output

* ROC-AUC compatible structural break scores ∈ \[0, 1]

---

## 4. Evaluation Protocol (ADIA-Aligned)

1. Apply the full pipeline to each training ID.
2. Extract topological metrics pre/post.
3. Train a shallow classifier or compute thresholded break score.
4. Use `y_train` for supervised evaluation.
5. Apply trained detector to `X_test` (submission-ready).

---

## 🔁 CIv6 System View

```text
Time Series
  └▶ Symbolic Encoding (delta/permutation)
       └▶ ECA Evolution (Rule 110)
            └▶ Transformer Attention Probing
                  ├▶ Loop Energy Analyzer
                  ├▶ Holonomy + Curvature Spectrum
                  ├▶ Entropy/FIM Divergence
                  └▶ Structural Break Scoring
```

---

## 🧪 Implementation Modules to Build

* `run_eca()` → Evolve bit sequences
* `extract_cycles_and_log_wilson()` → Detect closed attention paths
* `compute_holonomy_spectrum()` → Eigenvalues from Q/K loop transport
* `compare_pre_post_topology()` → Vector of topological metrics
* `structural_break_score()` → Final ROC-compatible scalar

---

## ✅ Ready for Prototyping

This proposal is now concretely aligned with:

* The ADIA Lab challenge schema
* Transformer latent state observables
* Geometric-topological signal tracking
* CIv6 cybernetic consistency modeling

It is fully modular and ready to be integrated into the ADIA baseline notebook.
