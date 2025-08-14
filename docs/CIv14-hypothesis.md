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

### **Substrate Variants**

#### **Symbolic Substrate**

* **Mechanistic Role:**

  * Encodes discrete causal structures through compressive motifs, capturing minimal generative programs.
  * Functions as a model-agnostic detection layer for algorithmic, topological, and semantic regime shifts.
* **Key Signals:**

  * **φ (Program Length):** Measures the length of the minimal program required to generate a sequence. Shorter programs indicate higher compressibility and potential for generalization.
  * **Δφ (Program Length Change):** Tracks changes in program length over time, signaling shifts in the underlying generative process.
  * **Divergence Metrics:** Quantifies the divergence between symbolic representations and latent encodings, indicating structural breaks or regime changes.
* **Integration with Other Substrates:**

  * Provides interpretable outputs (e.g., φ-scored sketches) that inform and guide the adaptation of latent models.
  * Works in tandem with the latent substrate to detect and explain regime shifts, facilitating a comprehensive understanding of system dynamics.

#### **Latent Substrate**

* **Mechanistic Role:**

  * Captures compressed dynamics of sequences, encoding fluid semantic continuity via distributed representations.
  * Monitors structural complexity and stability, identifying anomalies or shifts in the data stream.
* **Key Signals:**

  * **ΔBDM (Bayesian Description Length Change):** Measures changes in the description length of the model's posterior distribution, indicating shifts in data structure.
  * **ΔMDL (Minimum Description Length Change):** Assesses changes in the model's ability to compress data, reflecting the emergence of new patterns or regimes.
  * **Forecast Error (MSE):** Evaluates the accuracy of predictions, with increased error signaling potential regime changes.
* **Integration with Other Substrates:**

  * Provides statistical insights that complement symbolic representations, enhancing the system's ability to detect and adapt to structural changes.
  * Collaborates with the symbolic substrate to generate and evaluate potential explanations for observed shifts.

#### **Unified Substrate**

* **Mechanistic Role:**

  * Integrates symbolic and latent representations to form a cohesive understanding of system dynamics.
  * Facilitates the forecasting of future states and the generation of interpretable outputs for external agents.
* **Key Signals:**

  * **Regime Probability:** Combines ΔBDM, ΔMDL, and divergence metrics to assess the likelihood of a regime shift.
  * **Forecasted Sequences:** Generated through decompression of symbolic sketches, providing potential future trajectories.
  * **φ-Scored Outputs:** Represents the interpretability and quality of generated outputs, guiding stakeholder interactions.
* **Integration with Other Substrates:**

  * Serves as the interface between internal processing and external communication, ensuring that the system's outputs are both accurate and understandable.
  * Enables a feedback loop where external inputs can influence internal representations, promoting adaptive behavior.

---


