# **Substrate Mapping Across Hypotheses**

| Version   | Symbolic Substrate                                                                                                                        | Latent Substrate                                                                           | Unified / Integration Layer                                                                                                                | Notes / Mechanistic Role                                                                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CIv13** | ✔ Captures discrete rules and abstract patterns from sequences; symbolic encoder extracts regime-specific motifs                          | ✔ Latent encoder captures compressed representations of sequence dynamics                  | ✖ Minimal integration: divergence between symbolic and latent streams signals structural breaks                                            | Focused on **detection**; divergence (symbolic vs latent) is the measurable signal; no forecasting or adaptation                                                                                  |
| **CIv14** | ✔ Symbolic encoder continues to extract minimal generative programs; outputs φ-scored sketches                                            | ✔ Latent encoder continues to capture compressed dynamics                                  | ✔ **Decompression/forecaster integrates symbolic and latent streams**; ΔBDM and ΔMDL combine with divergence to produce regime probability | Focus shifts from detection → **dynamic modeling**; system uses both substrates to **forecast** next sequences and explain regime changes; interpretable symbolic outputs exposed to stakeholders |
| **CIv15** | ✔ Symbolic programs become **self-editing**, stored in versioned program library; supports causal perturbation and counterfactual testing | ✔ Latent encoder continues to capture compressed trends; used for divergence + forecasting | ✔ **Autopoietic loop unifies symbolic + latent + action evaluation**; controller edits programs based on ΔMDL, φ, and utility              | Focus moves from dynamic viability → **self-maintaining, planning, open-ended learning**; full neurosymbolic autopoietic integration; actionable outputs for downstream systems                   |

Perfect — let’s map **CIv13 → CIv14 → CIv15** onto the **substrates** (symbolic, latent, unified) so you can clearly see how each hypothesis incrementally engages each representation layer. I’ll structure it rigorously, mirroring CIv13 style.

---

### **Key Observations**

1. **Progression from CIv13 → CIv15**

   * CIv13: Divergence detection, substrates separate.
   * CIv14: Forecasting & dynamic adaptation, substrates partially unified via decompressor.
   * CIv15: Self-maintenance and planning, full autopoietic loop unifies symbolic, latent, and decision-making.

2. **Substrate Roles**

   * **Symbolic:** abstraction, minimal program encoding, human/interpretable outputs.
   * **Latent:** compressed patterns, statistical dynamics, anomaly/shift detection.
   * **Unified:** combination of symbolic foresight + latent representation + internal controller/planning → autopoietic viability.

3. **Mechanistic Metrics Across Substrates**

   * **Symbolic:** φ-metric, program length, edit acceptance.
   * **Latent:** ΔBDM, ΔMDL, forecast MSE.
   * **Unified:** regime probability, autopoietic viability, action utility.

---
