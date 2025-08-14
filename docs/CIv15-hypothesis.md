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

If you want, I can now **draw a CIv14 → CIv15 modular diagram** with all encoders, monitors, synthesis heads, decompressor, controller loops, and program library flows fully annotated for implementation. This would mirror the CIv13 diagram style.

Do you want me to do that next?
