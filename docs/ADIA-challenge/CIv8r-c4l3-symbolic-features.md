# C4 Level 3: Component View – `symbolic_features.py`

This component view describes the internal structure of the `symbolic_features.py` module in the CIv8r mesoscope system. It is responsible for extracting symbolic features from univariate time series using cellular automata (ECA), symbolic tokenization, and motif-level analysis.

---

## 📦 Container: `symbolic_features.py`

**Purpose:** Generate symbolic representations and features from pre-break and post-break segments of a time series for downstream fusion and scoring.

**Source Repositories:**

* `TransformerECA` (`TransformerECA_ipynb.txt`)
* `Intelligence_at_the_edge_of_chaos` (`Intelligence_at_the_edge_of_chaos_py.txt`)

**GitHub Traceback License:**

* Apache 2.0 (confirmed for both uploaded sources)

---

## 🧩 Internal Components

| Component                                          | Description                                                               | GitHub Source                                                                   | Reuse Type |
| -------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| `generate_symbolic_sequence(series, rule, steps)`  | Converts raw time series into symbolic bit string using ECA dynamics      | `cellular_automaton()` in `utils.py`【66†Intelligence\_at\_the\_edge\_of\_chaos】 | 🛠️ Adapt  |
| `tokenize_symbolic_sequence(symbolic_sequence)`    | Applies 1D tokenizer to generate motif tokens                             | `SimpleTokenizer`【65†TransformerECA\_ipynb】                                     | ♻️ Reuse   |
| `compute_entropy_delta(pre_tokens, post_tokens)`   | Measures change in symbolic entropy across boundary                       | New logic (entropy stats)                                                       | ✳️ Build   |
| `compute_compression_delta(pre_seq, post_seq)`     | Uses LZ or dictionary compression size delta                              | Adapted from motif stack ideas                                                  | ✳️ Build   |
| `extract_motif_frequencies(tokens)`                | Extracts motif frequency table from token list                            | Implicit in GPT2-style prep【66†Intelligence\_at\_the\_edge\_of\_chaos】          | 🛠️ Adapt  |
| `compute_motif_kl_divergence(pre_freq, post_freq)` | Computes KL divergence or JS divergence between pre/post frequency tables | New logic                                                                       | ✳️ Build   |
| `generate_feature_vector()`                        | Combines all symbolic metrics into output dict                            | New code                                                                        | ✳️ Build   |

---

## 🔗 Dependencies

* `numpy`, `collections.Counter`, `math.log2`
* Optional: `lzma` or custom LZ complexity estimator
* `tqdm`, `re` for tokenizer sanitation (from TransformerECA)

---

## 🧪 Test Strategy

| Component                       | Test Case                                                       |   |           |   |                              |
| ------------------------------- | --------------------------------------------------------------- | - | --------- | - | ---------------------------- |
| `generate_symbolic_sequence()`  | Compare output bits for known ECA rules (e.g. Rule 30, Rule 90) |   |           |   |                              |
| `tokenize_symbolic_sequence()`  | Validate consistent tokens from symbolic strings                |   |           |   |                              |
| `compute_entropy_delta()`       | Assert entropy(pre) < entropy(post) for injected drift          |   |           |   |                              |
| `compute_motif_kl_divergence()` | KL(p                                                            |   | q) ≠ KL(q |   | p); verify symmetry using JS |

---

## 📤 Output Format

```python
{
    'entropy_delta': 0.23,
    'compression_delta': 0.18,
    'kl_divergence': 1.09,
    'js_divergence': 0.65,
    'pre_freq': {"010": 8, "110": 3},
    'post_freq': {"010": 2, "001": 6}
}
```

This dictionary will be passed to the `fusion_score.py` module for integration with latent features.

---

## ✅ Status Summary

* Core symbolic engine fully backed by uploaded GitHub repos
* Divergence and delta metrics need to be implemented
* All components Colab-compatible and checkpointable

---

**Next Steps:**

* Integrate this component view into CIv8r internal whitepaper as Appendix C
* Begin prototyping symbolic runtime in Colab with minimal inputs
* Validate outputs on ADIA sample test data

---

**End of Component View**
# C4 Level 3: Component View – `symbolic_features.py`

This component view describes the internal structure of the `symbolic_features.py` module in the CIv8r mesoscope system. It is responsible for extracting symbolic features from univariate time series using cellular automata (ECA), symbolic tokenization, and motif-level analysis.

---

## 📦 Container: `symbolic_features.py`

**Purpose:** Generate symbolic representations and features from pre-break and post-break segments of a time series for downstream fusion and scoring.

**Source Repositories:**

* `TransformerECA` (`TransformerECA_ipynb.txt`)
* `Intelligence_at_the_edge_of_chaos` (`Intelligence_at_the_edge_of_chaos_py.txt`)

**GitHub Traceback License:**

* Apache 2.0 (confirmed for both uploaded sources)

---

## 🧩 Internal Components

| Component                                          | Description                                                               | GitHub Source                                                                   | Reuse Type |
| -------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| `generate_symbolic_sequence(series, rule, steps)`  | Converts raw time series into symbolic bit string using ECA dynamics      | `cellular_automaton()` in `utils.py`【66†Intelligence\_at\_the\_edge\_of\_chaos】 | 🛠️ Adapt  |
| `tokenize_symbolic_sequence(symbolic_sequence)`    | Applies 1D tokenizer to generate motif tokens                             | `SimpleTokenizer`【65†TransformerECA\_ipynb】                                     | ♻️ Reuse   |
| `compute_entropy_delta(pre_tokens, post_tokens)`   | Measures change in symbolic entropy across boundary                       | New logic (entropy stats)                                                       | ✳️ Build   |
| `compute_compression_delta(pre_seq, post_seq)`     | Uses LZ or dictionary compression size delta                              | Adapted from motif stack ideas                                                  | ✳️ Build   |
| `extract_motif_frequencies(tokens)`                | Extracts motif frequency table from token list                            | Implicit in GPT2-style prep【66†Intelligence\_at\_the\_edge\_of\_chaos】          | 🛠️ Adapt  |
| `compute_motif_kl_divergence(pre_freq, post_freq)` | Computes KL divergence or JS divergence between pre/post frequency tables | New logic                                                                       | ✳️ Build   |
| `generate_feature_vector()`                        | Combines all symbolic metrics into output dict                            | New code                                                                        | ✳️ Build   |

---

## 🔗 Dependencies

* `numpy`, `collections.Counter`, `math.log2`
* Optional: `lzma` or custom LZ complexity estimator
* `tqdm`, `re` for tokenizer sanitation (from TransformerECA)

---

## 🧪 Test Strategy

| Component                       | Test Case                                                       |   |           |   |                              |
| ------------------------------- | --------------------------------------------------------------- | - | --------- | - | ---------------------------- |
| `generate_symbolic_sequence()`  | Compare output bits for known ECA rules (e.g. Rule 30, Rule 90) |   |           |   |                              |
| `tokenize_symbolic_sequence()`  | Validate consistent tokens from symbolic strings                |   |           |   |                              |
| `compute_entropy_delta()`       | Assert entropy(pre) < entropy(post) for injected drift          |   |           |   |                              |
| `compute_motif_kl_divergence()` | KL(p                                                            |   | q) ≠ KL(q |   | p); verify symmetry using JS |

---

## 📤 Output Format

```python
{
    'entropy_delta': 0.23,
    'compression_delta': 0.18,
    'kl_divergence': 1.09,
    'js_divergence': 0.65,
    'pre_freq': {"010": 8, "110": 3},
    'post_freq': {"010": 2, "001": 6}
}
```

This dictionary will be passed to the `fusion_score.py` module for integration with latent features.

---

## ✅ Status Summary

* Core symbolic engine fully backed by uploaded GitHub repos
* Divergence and delta metrics need to be implemented
* All components Colab-compatible and checkpointable

---

**Next Steps:**

* Integrate this component view into CIv8r internal whitepaper as Appendix C
* Begin prototyping symbolic runtime in Colab with minimal inputs
* Validate outputs on ADIA sample test data

---

**End of Component View**
