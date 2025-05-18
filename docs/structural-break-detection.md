**Research Proposal: Detecting Structural Breaks in Time Series using Permutation Entropy and Algorithmic Generative Modeling**

**1. Introduction & Motivation**

Structural breaks, or abrupt changes in the underlying data-generating process of a time series, are critical to detect across various domains like finance, climatology, and industrial monitoring. Traditional statistical methods often focus on specific types of changes (e.g., mean, variance) and may make strong assumptions about the data. Algorithmic and information-theoretic approaches offer a model-agnostic way to capture changes in complexity and dynamics. This proposal outlines a two-part investigation:
*   **Part A:** A robust baseline approach using Permutation Entropy (PE) to quantify changes in complexity around a specified boundary point.
*   **Part B:** An advanced approach extending PE-derived encodings to infer changes in underlying generative rules using Elementary Cellular Automata (ECA) dynamics, potentially accelerated by Transformer models.

The goal is to develop a methodology that not only detects structural breaks with high accuracy (as measured by ROC AUC) but also offers insights into the nature of the change by attempting to identify shifts in the "causal rules" governing the time series.

**2. Overall Objective**

To develop and evaluate novel methods for structural break detection in univariate time series by:
1.  Quantifying changes in time series complexity using Permutation Entropy.
2.  Inferring changes in the underlying generative mechanisms by modeling time series segments with Elementary Cellular Automata, using PE-derived binary encodings.

**3. Part A: Structural Break Detection via Permutation Entropy Change**

**3.1. Rationale**
Permutation Entropy (PE) is a complexity measure sensitive to the ordinal patterns in a time series. It is robust to noise, independent of signal amplitude, and can be reliably estimated from relatively short data segments. A significant change in PE across a boundary point suggests a change in the underlying dynamics of the time series, indicative of a structural break.

**3.2. Methodology**
For each time series provided in the dataset (`X_train` for development/training, `X_test` for inference):

   **3.2.1. Data Segmentation:**
   Divide the time series into two segments based on the `period` indicator:
   *   `segment_before`: Data where `period == 0`.
   *   `segment_after`: Data where `period == 1`.

   **3.2.2. Permutation Entropy Calculation:**
   For both `segment_before` and `segment_after`:
   *   Calculate PE using a standard algorithm (e.g., Bandt & Pompe, 2002).
   *   Hyperparameters:
        *   Embedding dimension (`m`): Explore values (e.g., 3 to 7).
        *   Time delay (`tau`): Explore values (e.g., 1 to 3).
   *   This will yield `PE_before` and `PE_after`.

   **3.2.3. Feature Engineering for Classification:**
   Construct features based on the PE values:
   *   `delta_PE_abs = abs(PE_after - PE_before)`
   *   `delta_PE_norm = (PE_after - PE_before) / ( (PE_after + PE_before) / 2 + epsilon )` (normalized difference)
   *   Optionally, include `PE_before` and `PE_after` directly.
   *   If multiple (`m`, `tau`) pairs are used, generate corresponding features for each.
   *   Consider adding other statistical features as identified in the initial competition baseline (mean, std dev differences) for comparison and potential combination.

   **3.2.4. Model Training (`train` function):**
   *   Extract the PE-derived features (and any other chosen features) for all series in `X_train`.
   *   Train a binary classifier (e.g., RandomForestClassifier, GradientBoostingClassifier, Logistic Regression) using these features to predict `y_train` (the ground truth structural break labels).
   *   Perform hyperparameter tuning for the classifier and PE parameters (`m`, `tau`) using cross-validation on `X_train`.

   **3.2.5. Prediction (`infer` function):**
   *   For each incoming test series:
        *   Perform segmentation.
        *   Calculate `PE_before` and `PE_after` using the optimal (`m`, `tau`) found during training.
        *   Construct the feature vector.
        *   Use the trained classifier to output a probability score (between 0 and 1) indicating the likelihood of a structural break.

**3.3. Expected Outcome & Evaluation (Part A)**
*   A trained model capable of predicting structural breaks based on changes in PE.
*   Performance evaluated using ROC AUC score.
*   Analysis of feature importance to understand the contribution of PE-based features.
*   This part serves as a strong, interpretable baseline built on robust complexity measures.

**4. Part B: Inferring Changes in Generative Rules via PE-Encoded ECA Modeling**

**4.1. Rationale**
While Part A detects *if* a break occurred based on complexity change, Part B aims to delve deeper by hypothesizing that the time series dynamics can be modeled by Elementary Cellular Automata (ECA). A structural break would then correspond to a change in the underlying ECA rule(s) governing the system. By encoding the time series segments into binary strings using permutation patterns, we can attempt to identify these generative rules and their changes.

**4.2. Methodology**

   **4.2.1. PE-based Binary Encoding:**
   For `segment_before` and `segment_after` of each time series:
   *   As discussed:
        1.  Choose an embedding dimension `m_eca` (distinct from `m` in Part A, or could be the same) and delay `tau_eca`.
        2.  Identify the sequence of ordinal permutation patterns in the segment.
        3.  Assign a unique binary code (length `k = ceil(log2(m_eca!))`) to each of the `m_eca!` possible patterns.
        4.  Concatenate these binary codes to form `binary_string_before` and `binary_string_after`.

   **4.2.2. ECA Rule Inference / Modeling (Iterative Development):**

   **Approach 1: Algorithmic Complexity of Encoded Strings (Simpler, Faster for Initial Tests)**
   *   Calculate the algorithmic complexity (e.g., using Block Decomposition Method (BDM) or Lempel-Ziv compression length) of `binary_string_before` and `binary_string_after`.
   *   Features for the classifier (from Part A) would include `abs(BDM_after - BDM_before)`, etc., for these PE-derived binary strings.
   *   This approach checks if the *complexity of the PE-encoded dynamics* changes.

   **Approach 2: ECA Rule Search with Genetic Algorithm (Your Capstone Inspired)**
   *   For `binary_string_before`: Use a Genetic Algorithm (GA) to find the "best" ECA rule (or rule tuple, e.g., single or double rule as in your capstone) that, when using `binary_string_before` as an initial condition, generates an evolution whose BDM or other algorithmic distance to `binary_string_before` (or its MILS-compressed version) is minimized. Let this be `rule_tuple_before`.
   *   Repeat for `binary_string_after` to find `rule_tuple_after`.
   *   **Features for Classifier:**
        *   A distance metric between `rule_tuple_before` and `rule_tuple_after` (e.g., Hamming distance if rules are represented as numbers/bitstrings, or a more sophisticated behavioral distance).
        *   Boolean feature: `is_rule_tuple_same`.
        *   Complexity of the rules themselves (e.g., BDM of the rule's binary representation).
   *   *Challenge:* High computational cost in `infer`.

   **Approach 3: ECA Rule Inference with Pre-trained Transformer (Burtsev Inspired)**
   *   **Offline Pre-training:** Train an "ECA-Transformer" (as per Burtsev, 2023) on a large synthetic dataset of ECA evolutions (using *raw binary strings* as ECA states, not necessarily PE-encoded strings initially, though training on PE-encoded ECA-like dynamics is an advanced option). This model learns to infer ECA rules from orbits (e.g., Burtsev's O-SR task).
   *   **Feature Extraction (in `train`/`infer`):**
        1.  Feed `binary_string_before` (the PE-encoded string) to the pre-trained ECA-Transformer.
        2.  The Transformer outputs an `inferred_rule_representation_before` (this could be the rule's binary string or an embedding).
        3.  Repeat for `binary_string_after` to get `inferred_rule_representation_after`.
        4.  **Features for Classifier:** Similar to Approach 2, based on comparing these inferred rule representations.
   *   *Advantage:* Significantly faster inference than GA.
   *   *Consideration:* The Transformer is trained on general ECA dynamics. Its ability to interpret PE-encoded strings as "ECA-like" orbits needs validation. The PE-encoding effectively creates a new type of discrete dynamical system.

   **4.2.3. Model Training and Prediction (Part B):**
   *   Similar to Part A, train a classifier using the features derived from the chosen ECA modeling approach (Approach 1, 2, or 3).
   *   The target remains `y_train`.
   *   Prediction in `infer` uses the same feature extraction and trained classifier.

**4.3. Expected Outcome & Evaluation (Part B)**
*   Models that attempt to link structural breaks to changes in inferred generative ECA rules.
*   Comparison of performance (ROC AUC) between Part A and the different approaches in Part B.
*   Insights into whether specific types of rule changes (if identifiable) correlate with structural breaks.
*   If successful, this part offers a deeper, more causal understanding of the breaks.

**5. General Implementation Details**

*   **Libraries:** `pandas`, `numpy`, `scipy`, `scikit-learn`. For PE, libraries like `nolds` or `antropy` (or custom implementation). For ECA/BDM, potentially custom code or specialized libraries (e.g., `pybdm`, `CellPyLib`). For Transformers, `pytorch` or `tensorflow`.
*   **Parameter Tuning:** Rigorous cross-validation for all hyperparameters (PE: `m`, `tau`; PE-ECA: `m_eca`, `tau_eca`; classifier parameters; GA parameters if used).
*   **Benchmarking:** Compare against the competition baseline and potentially other standard structural break detection methods (e.g., Chow test, CUSUM if applicable, though they make more assumptions).

**6. Potential Challenges & Mitigation**

*   **Computational Cost (Part B, Approach 2):** GA for rule search is slow. Mitigation: Start with simpler rule sets, parallelize, or move to Approach 3 (Transformer).
*   **Hyperparameter Sensitivity:** PE and ECA modeling have key parameters. Mitigation: Systematic search and cross-validation.
*   **Interpretation of "Rules" (Part B):** Inferred ECA rules from complex, noisy real-world data (even after PE-encoding) might be abstract. Focus on *changes* in these inferred rules rather than their absolute interpretability initially.
*   **PE-Encoding for ECA:** The mapping from continuous time series to permutation patterns and then to binary strings is a form of information transformation. The dynamics of these PE-encoded strings might behave differently from canonical ECAs. This is an area of exploration.

**7. Timeline (Illustrative)**
*   Phase 1 (Weeks 1-2): Implement Part A, establish robust PE-based baseline.
*   Phase 2 (Weeks 3-5): Implement PE-to-binary encoding. Explore Part B, Approach 1 (BDM on PE-encoded strings).
*   Phase 3 (Weeks 6-8): If time/resources permit, explore Part B, Approach 3 (ECA-Transformer). Approach 2 (GA) might be too slow for competition but valuable for research.
*   Phase 4 (Week 9-10): Final model selection, testing, documentation.

**8. Conclusion**

This research proposal outlines a structured approach to tackling the structural break detection challenge, progressing from a robust complexity-based method (PE) to a more ambitious generative modeling approach (PE-ECA). The findings will contribute to understanding the efficacy of these techniques for analyzing complex time series and potentially offer deeper insights into the nature of structural changes.

---
