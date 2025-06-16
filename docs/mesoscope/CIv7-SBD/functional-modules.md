Below is a **CIv7-SBD Architecture Module Table** for the **Structural Break Detection** domain. It aligns with your style from CIv7-TI and integrates:

* **Modular function blocks**
* **Function signatures**
* **Input/output specs**
* **Key research dependencies**
* **Layer mapping (L1–L3)**

---

### ✅ **CIv7-SBD Architecture Module Table**

| **Module ID** | **Module Name**                       | **Function Signature**                                                              | **Input**                                           | **Output**                   | **Layer** | **Foundational Dependencies**                                                                          |
| ------------- | ------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| `SBD-L1A`     | Statistical Divergence Scorer         | `def compute_ttest_score(df: pd.DataFrame) -> float`                                | `df`: time series with `value` and `period` columns | t-test-based score           | L1        | ROC AUC theory, Welch’s t-test, Grundy et al. (2025) – Forecast Error-Based Changepoint Detection      |
| `SBD-L1B`     | Symbolic Compression Divergence       | `def symbolic_diff(df: pd.DataFrame) -> float`                                      | Discretized symbolic sequence from `value` column   | Compression delta            | L1        | Zenil et al., Riedel & Zenil (Rule Primality), OpenThoughts (2024), BrightStar Labs (2025)             |
| `SBD-L1C`     | Geometric Summary Shift               | `def latent_shift(df: pd.DataFrame) -> float`                                       | `value` column split by `period`                    | Sum of mean/std deltas       | L1        | Jha et al. (2024), Walch (2024), Zhang et al. – Edge of Chaos, Hodge et al. – Geometry in Transformers |
| `SBD-L1D`     | Combined Breakpoint Score             | `def break_score(df: pd.DataFrame) -> float`                                        | Outputs from `L1A`, `L1B`, `L1C`                    | Final break prediction score | L1        | Ensemble signal aggregation, CIv7-ECA hypothesis                                                       |
| `SBD-L2A`     | Feature Extractor for ML Classifier   | `def extract_features(df: pd.DataFrame) -> Dict[str, float]`                        | Time series `df` with boundary period marking       | Dict of engineered features  | L2        | Grünwald (MDL), Peter Grünwald & Roos – MDL Revisited, AlphaEvolve (DeepMind), Ha & Schmidhuber        |
| `SBD-L2B`     | ML Classifier for Break Prediction    | `def train_classifier(X_train: List[Dict], y_train: List[int]) -> Model`            | Feature dicts + labels                              | Trained model                | L2        | Grosse et al. – Occam-Razor Deep Models, Chen et al. – SASR, RD-Agent(Q)                               |
| `SBD-L2C`     | Model-Based Inference Pipeline        | `def infer_breaks(model, X_test: List[pd.DataFrame]) -> List[float]`                | Trained model + test data list                      | List of prediction scores    | L2        | AlphaEvolve (2025), OpenThoughts – Symbolic Inference Recipes                                          |
| `SBD-L3A`     | Attribution Shift Analyzer            | `def track_attribution_shift(attn_now: np.ndarray, attn_base: np.ndarray) -> float` | Attention matrices before/after boundary            | Attribution drift score      | L3        | Sakabe et al., Anthropic Circuit Tracer (2025), Sutskever – Joint Encoding Failures                    |
| `SBD-L3B`     | Semantic Compression Anomaly Detector | `def detect_semantic_anomaly(encoded_seq: List[Any]) -> float`                      | LLM or latent representations of token sequences    | Semantic drift likelihood    | L3        | Shani et al. (2024), Walch (torsion), Zhang et al. (Platonic Drift), CIv7-LLM Hypothesis               |
| `SBD-UTIL`    | Visualization + Debug Tools           | `def plot_series_with_boundary(df: pd.DataFrame, id: int) -> None`                  | Raw series with boundary point                      | Matplotlib plot              | Utility   | For inspection only; optional                                                                          |

---

### 📌 **Layer Mapping (CIv7 Style)**

* **L1: Break Signal Generators**

  * Direct signal scoring methods (t-test, compression delta, summary stat shift).
  * No model training required—fully interpretable.

* **L2: Model-Based Structural Fault Estimation**

  * Learns patterns in symbolic/engineered break indicators.
  * Supports generalization beyond local features.

* **L3: Latent Drift & Attribution Trace Detectors**

  * Attribution-path and embedding-level indicators.
  * Detects "hidden breaks" not visible in surface-level signals.

---

### 🧠 Next Steps

Would you like the table exported as:

* CSV, Markdown, or PDF for documentation?
* Incorporated into the onboarding guide?
* Turned into a visual architecture diagram (e.g., via Mermaid or system schematic)?
* Auto-generated code stubs for each module?

Let me know your preference.
