# **CIv14 – Compression‑Driven Dynamic Viability**

### **Hypothesis Statement**

CIv14 extends CIv13’s symbolic-latent divergence detection into a **self-monitoring cybernetic loop**, where regime transitions are modeled as **shifts in minimal generative programs**. These shifts are detected via **ΔBDM** and **ΔMDL** signals, forecasted via **decompression of symbolic sketches**, and communicated to external agents via **φ-scored program outputs**. CIv14 thus operationalizes **dynamic viability**: the system maintains internal consistency while adapting to structural changes in real time.

---

### **Principles**

1. **Dynamic Regime Modeling:**

   * Regime change is defined as a statistically significant change in ΔBDM (≥ threshold θ\_BDM) and ΔMDL (≥ θ\_MDL) over a time window τ.
   * Temporal structure is captured via recursive compression → decompression cycles.

2. **Compression–Prediction Equivalence:**

   * Forecasting quality is strictly proportional to compressibility:
     $\text{Prediction Accuracy} \propto \text{ΔCompression Gain (ΔBDM)}$

3. **Interpretability via Symbolic Regression:**

   * Sketch outputs are scored with φ-metric (Non-Print & Non-Ordinal > Ordinal > Print).
   * Only φ ≥ φ\_min outputs are communicated to stakeholders.

4. **Neurosymbolic Integration:**

   * Latent encoder (Transformer/TSEncoder) captures statistical dynamics.
   * Symbolic encoder captures discrete rules for minimal generative programs.
   * Divergence between streams signals regime shifts.

---

### **Mechanism**

* **Symbolic Encoder:**

  * Inputs: permutation/binary sequences of dimension d, delay τ.
  * Outputs: symbolic embeddings for minimal generative pattern detection.
* **Latent Encoder:**

  * Inputs: raw sequence segments.
  * Outputs: compressed latent vectors; used for divergence calculation.
* **Complexity Monitor:**

  * Compute ΔBDM per segment:
    $\Delta \text{BDM} = |\text{BDM}_{t+1} - \text{BDM}_t|$
  * Compute ΔMDL = |MDL\_{model+error,t+1} - MDL\_{model+error,t}|
  * If ΔBDM ≥ θ\_BDM or ΔMDL ≥ θ\_MDL → regime alert.
* **Program Synthesis Head:**

  * Generates candidate symbolic sketches.
  * φ scoring ensures quality and interpretability.
* **Decompressor Forecaster:**

  * Runs symbolic sketch forward to simulate next window.
  * Forecast compared to actual sequence → updates ΔMDL/ΔBDM.
* **Divergence Integration:**

  * Combine latent-symbolic divergence and ΔBDM for regime probability.
* **External Interface:**

  * Outputs φ-scored sketches and predicted sequence segments to analysts or agents.

---

### **Success Metrics**

* **Forecast Accuracy:** MSE between predicted and observed sequences.
* **Compression Efficiency:** Increase in ΔBDM per detected regime.
* **Interpretability:** Ratio of non-print φ outputs ≥ φ\_min.
* **Regime Detection Recall:** True positive rate for known regime shifts.
* **Internal Consistency:** Stabilization of ΔMDL after regime adaptation.

---

### **References**

1. Hernández-Espinosa et al., 2024 – *SuperARC*
2. Zenil et al., 2018 – *BDM/CTM complexity methods*
3. Riedel & Zenil, 2025 – *ECA rule minimality and causal decomposition*
4. Maturana & Varela, 1980 – *Autopoiesis*

---


