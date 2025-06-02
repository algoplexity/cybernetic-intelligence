**Solution Proposal: Adaptive, Algorithmically-Principled Structural Break Detection using an ECA-MILS-BDM-Transformer Framework**

**Version:** 1.0
**Date:** 2nd June 2025
**Challenge:** ADIA Lab Structural Break Challenge

**1. Introduction & Motivation**

This document outlines a novel solution for detecting structural breaks in univariate time series, specifically tailored for the ADIA Lab Structural Break Challenge. Current methods often rely on statistical assumptions or lack deep interpretability. Our approach is founded on the hypothesis that structural breaks represent fundamental shifts in the underlying algorithmic data-generating process. We propose to model these processes using Elementary Cellular Automata (ECAs), identify changes in optimal ECA rules using the Minimum Description Length (MDL) principle (approximated via Block Decomposition Method - BDM), and enhance this process with Minimal Information Loss Selection (MILS) for robust representation and a specialized Transformer for dynamic fidelity assessment and efficient rule exploration. An adaptive framework will manage computational resources.

**2. Overarching Hypothesis (Consolidated)**

The full hypothesis is available [here](https://algoplexity.github.io/cybernetic-intelligence/algorithmic-intelligence/structural-break-detection-hypothesisv2) - . It covers: ECA as generative model, MILS for representation, BDM-MDL for break adjudication, Transformer for fidelity/efficiency, Adaptive Application, and Interpretable Characterization.

**3. Core Methodology & Pipeline**

The solution will be implemented within the CrunchDAO `train()` and `infer()` structure.

**3.1. Offline Preparation (`train()` function components):**

*   **3.1.1. Optimal Binary Encoding Scheme Discovery (Experiment - see Section 4):**
    *   **Goal:** Identify the best initial binarization method (e.g., Sign of First Difference (S1D), Binarized Permutation Entropy, Binarized Volatility) and its parameters (e.g., `target_width` for 2D reshaping).
    *   **Process:** For each candidate encoding, apply it to `X_train` segments, then apply MILS, then assess using the `ECA_Dynamics_Transformer` (see 3.1.2) to see which encoding leads to MILS-processed patterns whose "ECA-ness" changes most discriminatively across known breaks. The primary metric is ROC AUC on `y_train`.
    *   **Output:** The chosen binary encoding function and its parameters (saved to `model_directory_path`).
*   **3.1.2. ECA Dynamics Transformer Training (Burtsev-style):**
    *   **Goal:** Train a Transformer (`ECA_Dynamics_Transformer`) proficient in predicting ECA evolutions and implicitly/explicitly inferring rules from 2D binary patterns.
    *   **Training Data:** Diverse set of ECA evolutions (various rules, initial conditions, possibly with added noise for robustness).
    *   **Architecture:** Suitable sequence-to-sequence or stateful Transformer architecture.
    *   **Output:** Trained `ECA_Dynamics_Transformer` model (saved to `model_directory_path`).
*   **3.1.3. MILS Configuration/Training (if applicable):**
    *   While MILS is GA-driven per instance, any global MILS parameters (e.g., GA settings for MILS, target compression ratio heuristics) determined during experimentation should be saved.
*   **3.1.4. Adaptive Framework Heuristics/Thresholds:**
    *   If the Tier 1 adaptive heuristic involves fixed thresholds or a simple trained model, these are prepared/trained here and saved.
*   **3.1.5. GA Configuration for ECA Rule Search:**
    *   Save default GA parameters (population size, mutation/crossover rates, termination criteria) to `model_directory_path`.

**3.2. Inference Pipeline (`infer()` function components - per test dataset):**

For each `dataset` in `X_test`:

*   **Step I: Adaptive Tier 1 Check (Optional, from Roadmap Phase 3):**
    1.  Apply fast heuristic(s) to `dataset` to get a preliminary break score.
    2.  If score is decisively low or high, `yield` corresponding prediction (e.g., 0.05 or 0.95) and skip to next dataset.
    3.  Else, proceed to Tier 2.

*   **Step II: Tier 2 - Full Algorithmic Analysis:**
    1.  **Data Preparation:**
        a.  Extract `continuous_pre`, `continuous_post`, and `continuous_full` segments from `dataset`.
        b.  Apply the **chosen optimal binary encoding scheme** (from 3.1.1) to get `B_raw_pre`, `B_raw_post`, `B_raw_full`.
    2.  **MILS Preprocessing (Fidelity to Thesis):**
        a.  `B_mils_pre = apply_mils(B_raw_pre, mils_config)`
        b.  `B_mils_post = apply_mils(B_raw_post, mils_config)`
        c.  `B_mils_full = apply_mils(B_raw_full, mils_config)`
        *   `apply_mils`: Implements the GA-based row/column selection to preserve algorithmic information (BDM-guided fitness).
    3.  **ECA Rule Discovery (GA or Transformer-Assisted GA):**
        a.  `Rule_A = find_optimal_eca(B_mils_pre, eca_ga_config, ECA_Dynamics_Transformer?)`
        b.  `Rule_B = find_optimal_eca(B_mils_post, eca_ga_config, ECA_Dynamics_Transformer?)`
        c.  `Rule_Global = find_optimal_eca(B_mils_full, eca_ga_config, ECA_Dynamics_Transformer?)`
        *   `find_optimal_eca`: GA searching for ECA rule.
            *   *Fitness Function:* Minimize `BDM_distance(B_mils_segment, MILS_Pattern(simulate_eca(Candidate_Rule)))`. `MILS_Pattern` means applying MILS also to the candidate ECA's raw output.
            *   The `ECA_Dynamics_Transformer` can optionally be used here to prune the GA search space or refine candidate rules.
    4.  **BDM-MDL Score Calculation:**
        a.  Simulate patterns from identified rules: `Pattern_A_raw = simulate_eca(Rule_A, ...)`, etc.
        b.  Apply MILS to generated patterns: `MILS_Pattern_A = apply_mils(Pattern_A_raw)`, etc.
        c.  Calculate BDM-based description length proxies (e.g., using BDM of XOR difference, or |BDM(obs) - BDM(gen)|):
            *   `L_A = BDM_cost(B_mils_pre, MILS_Pattern_A)`
            *   `L_B = BDM_cost(B_mils_post, MILS_Pattern_B)`
            *   `L_G = BDM_cost(B_mils_full, MILS_Pattern_Global)`
        d.  `MDL_Raw_Score = L_G - (L_A + L_B)` (Higher implies break).
    5.  **Transformer Fidelity Augmentation (Optional, from Roadmap Phase 2):**
        a.  `Fidelity_A_on_Post = ECA_Dynamics_Transformer.assess_fit(Rule_A, B_mils_post)` (or `B_raw_post`). This score indicates how well the pre-break rule explains post-break data dynamics. Low fidelity supports a break.
        b.  Combine `MDL_Raw_Score` with `Fidelity_A_on_Post` (e.g., weighted sum, multiplicative factor) to get `Final_Raw_Score`.
    6.  **Normalization & Yield:**
        a.  Normalize `MDL_Raw_Score` (or `Final_Raw_Score`) to a [0, 1] probability.
        b.  `yield normalized_prediction`.

**3.3. Post-Inference Analysis (for paper/interpretation):**

*   For detected breaks, apply **Causal Decomposition** to `Rule_A` and `Rule_B` to understand the nature of the change in algorithmic dynamics.

**4. Experiment for Optimal Binary Encoding Scheme Selection (Detailed)**

*   **Objective:** To determine which initial binarization method (S1D, binarized PE, etc.) and its parameters (e.g., `target_width` for 2D reshape) best prepares the data for the subsequent MILS, Transformer assessment, and ultimately, the ECA-MDL pipeline.
*   **Process:**
    1.  **Prerequisite:** Pre-train the `ECA_Dynamics_Transformer` (from 3.1.2).
    2.  **For each candidate binary encoding scheme (and its parameters):**
        a.  Take `X_train` segments (`continuous_pre`, `continuous_post`).
        b.  Apply current encoding: `B_raw_pre`, `B_raw_post`.
        c.  Apply MILS: `B_mils_pre`, `B_mils_post`.
        d.  **Transformer Assessment:**
            *   `Fit_Score_Pre = ECA_Dynamics_Transformer.assess_fit(B_mils_pre)` (how "ECA-like" is this MILS-processed pre-break pattern?)
            *   `Fit_Score_Post = ECA_Dynamics_Transformer.assess_fit(B_mils_post)`
        e.  `Delta_Fit = abs(Fit_Score_Post - Fit_Score_Pre)`.
        f.  Calculate ROC AUC of `Delta_Fit` against `y_train`.
    3.  **Decision:** Select the encoding scheme + parameters yielding the highest ROC AUC in this experiment. This chosen scheme is then used in the main pipeline (3.2.1.b).
*   **Rationale:** This uses the "smart" Transformer to quickly evaluate if an encoding (after MILS) produces binary patterns whose inherent "ECA-like structure" changes meaningfully across true breaks. This is faster than running the full GA for each encoding candidate.

**5. Key Modules & Responsibilities (as previously outlined):**

*   **Data Encoding & MILS Specialist:** Initial binarization, MILS implementation.
*   **ECA Modeling & GA Specialist:** GA for ECA rule search.
*   **Algorithmic Complexity, BDM-MDL & Scoring Specialist:** BDM implementation, MDL score logic, normalization.
*   **Transformer & Machine Learning Specialist:** `ECA_Dynamics_Transformer` training & integration, adaptive framework logic.
*   **Overall System Integration, Analysis & Interpretation Specialist:** System integration, main evaluations, causal decomposition, paper.

**6. Tools:**

*   `pandas`, `numpy`
*   `CellPyLib` (for ECA simulation)
*   `pybdm` (for BDM)
*   `antropy` (for Permutation Entropy if used in encoding)
*   `sklearn` (for ROC AUC, ML heuristics for Tier 1)
*   `pytorch`/`tensorflow`/`jax` (for Transformer)
*   `joblib` (for saving/loading models/configs)
*   `tqdm` (for progress bars)
*   `matplotlib`/`seaborn` (for plotting)

**7. Risk & Mitigation:**

*   **Computational Cost:** MILS and GA are intensive.
    *   *Mitigation:* Adaptive framework (Tier 1/2), efficient GA (caching, runtime limits, Transformer-guidance), subsetting `X_train` for experiments.
*   **MILS Complexity:** Implementing a full GA-driven MILS is a sub-project.
    *   *Mitigation:* Start with simpler MILS (e.g., fixed percentile of rows/columns based on BDM contribution heuristic) or defer full MILS if initial results without it (or with simplified MILS) are promising. (This would be a deviation from thesis detail).
*   **Transformer Training:** Training a robust `ECA_Dynamics_Transformer` requires a good dataset and architecture.
    *   *Mitigation:* Start with a smaller Transformer, use readily available ECA generation tools.
*   **Parameter Tuning:** Many components have parameters.
    *   *Mitigation:* Systematic experimentation, focus on most sensitive parameters first.
