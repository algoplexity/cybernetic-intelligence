# **CIv15 – Autopoietic Planner via Compression‑Aligned Self‑Evolution**

### **Hypothesis Statement**

CIv15 evolves CIv14 into a **self-maintaining, recursively optimizing system**. Minimal generative programs are **autonomously edited**, future outcomes are **simulated via decompression**, and actions are selected to **maximize compressibility and downstream utility**. CIv15 operationalizes **autopoietic planning** in a measurable cybernetic framework.

---

### **Principles**

1. **Autopoiesis:**

   * Program library L\_t evolves to maintain or reduce MDL while improving forecast skill.
   * ΔMDL ≤ 0 over rolling window indicates internal viability.

2. **Compression-Driven Planning:**

   * Candidate actions are evaluated via decompression forecast.
   * Utility function U = f(ΔBDM, forecast accuracy, domain-specific reward).

3. **Open-Ended Curriculum:**

   * Sequence complexity progressively increases: climber → random → domain-transfer.
   * System’s φ-scored sketch output tracks adaptability.

4. **Causal Robustness:**

   * Perturb symbolic programs; successful recovery measured by return to φ ≥ φ\_min and ΔMDL stabilization.

---

### **Mechanism**

* **Controller (Self-Editor & Planner):**

  * Proposes edits to program library; evaluates via ΔMDL + φ + U.
  * Accepts edits that improve forecast, compressibility, or utility.

* **Program Library:**

  * Stores minimal generative programs with versioning.
  * Supports causal perturbation testing and counterfactual evaluation.

* **Curriculum Engine:**

  * Generates sequences of increasing complexity for continual learning.
  * Supports domain transfer tasks (e.g., ECA, ADIA sequences, market motifs).

* **Simulator (Decompressor Forecaster):**

  * Rolls forward program sketches to evaluate candidate actions.
  * Outputs utility-maximizing sequence/action pairs.

* **Divergence Integration:**

  * Latent-symbolic divergence + program evolution signal → adaptive regime probability.

* **External Interface:**

  * Provides actionable, φ-scored symbolic forecasts to downstream agents.

---

### **Success Metrics**

* **Autopoietic Viability:** ΔMDL ≤ 0 with forecast MSE improving.
* **Compression Efficiency:** ΔBDM positive after self-edit cycles.
* **Action Utility:** Selected action maximizes U = f(ΔBDM, forecast accuracy, domain reward).
* **Adaptability:** Non-print φ outputs ≥ φ\_min across curriculum levels.
* **Robust Recovery:** Successful post-perturbation recovery to stable ΔMDL/ΔBDM.

---

### **References**

1. Hernández-Espinosa et al., 2024 – *SuperARC*
2. Zenil et al., 2018 – *BDM/CTM complexity methods*
3. Riedel & Zenil, 2025 – *ECA rule minimality and causal decomposition*
4. Maturana & Varela, 1980 – *Autopoiesis*
5. Ashby, 1956 – *Design for a Brain*
6. Burtsev et al., 2023 – *Learning rules at the edge of chaos*

---

### **Substrate Variants**

#### **Symbolic Substrate**

* **Mechanistic Role:**

  * Represents minimal generative programs that describe the system's behavior.
  * Supports self-editing and adaptation through the program library, enabling the system to modify its own structure in response to environmental changes.
* **Key Signals:**

  * **φ (Program Length):** Indicates the complexity of the symbolic representation; shorter programs are preferred for efficiency.
  * **Δφ (Program Length Change):** Tracks modifications in program length, reflecting adaptations to new regimes.
  * **Edit Acceptance Metrics:** Evaluates the success of proposed edits based on improvements in compressibility, utility, and forecast accuracy.
* **Integration with Other Substrates:**

  * Provides the foundational structure for the system's behavior, guiding the generation of latent representations and informing forecasting models.
  * Interacts with the latent substrate to ensure that adaptations are consistent with observed data patterns.

#### **Latent Substrate**

* **Mechanistic Role:**

  * Encodes the statistical properties of sequences, capturing underlying patterns and dynamics.
  * Assesses the impact of symbolic edits on data compression and forecasting accuracy.
* **Key Signals:**

  * **ΔBDM (Bayesian Description Length Change):** Measures the effect of symbolic edits on the model's posterior distribution.
  * **ΔMDL (Minimum Description Length Change):** Assesses changes in the model's ability to compress data, indicating the efficiency of symbolic adaptations.
  * **Forecast Error (MSE):** Evaluates the accuracy of predictions post-edit, guiding further adaptations.
* **Integration with Other Substrates:**

  * Provides feedback on the effectiveness of symbolic edits, informing the program library's evolution.
  * Collaborates with the symbolic substrate to ensure that adaptations lead to improved system performance.

#### **Unified Substrate**

* **Mechanistic Role:**

  * Orchestrates the integration of symbolic and latent representations to facilitate adaptive behavior.
  * Evaluates candidate actions based on decompression forecasts, selecting those that maximize utility and align with system goals.
* **Key Signals:**

  * **Utility Function (U):** Combines ΔBDM, ΔMDL, and forecast accuracy to evaluate the desirability of candidate actions.
  * **Action Selection Metrics:** Assesses the success of chosen actions based on their impact on system performance and adaptability.
  * **Regime Adaptation Indicators:** Monitors the system's ability to adjust to new regimes, ensuring continued viability.
* **Integration with Other Substrates:**

  * Serves as the decision-making center, integrating inputs from symbolic and latent substrates to select optimal actions.
  * Ensures that adaptations are coherent and lead to improved system performance, maintaining internal consistency and external utility.

---




