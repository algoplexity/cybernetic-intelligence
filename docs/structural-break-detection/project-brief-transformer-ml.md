**Project Brief: Development of ECA Dynamics Transformer for Structural Break Detection**

**Project Lead(s):** [Name(s) of Transformer & ML Specialist(s)]
**Date:** May 27, 2024 (Placeholder)
**Related to Overall Project:** Adaptive, Algorithmically-Principled Structural Break Detection (ADIA Challenge)

**1. Project Goal:**

To develop, train, and integrate a Transformer model (`ECADynamicsTransformer`) capable of understanding and predicting the evolution of Elementary Cellular Automata (ECAs). This model will serve two primary functions within the larger structural break detection system:
    a.  **Primary Tool for "Optimal Binary Encoding Scheme Discovery":** Assessing the "ECA-likeness" and predictability of MILS-processed binary patterns derived from different initial time series binarization methods.
    b.  **Augmentation for Final Break Score (Fidelity Check):** Providing a dynamics-aware score indicating how well an ECA rule (found by the main GA for a pre-break segment) can explain the dynamics of a post-break data segment.

**2. Core Responsibilities & Tasks:**

*   **Task 2.1: Implementation & Adaptation of Burtsev's TransformerECA Model:**
    *   **Action:** Clone and thoroughly study the `burtsev/TransformerECA` GitHub repository.
    *   **Action:** Implement (or adapt directly) the Transformer architecture, likely based on `BertForMaskedLM` as used in their "Orbit-State (O-S)" task.
    *   **Key Considerations:** Pay close attention to their input tokenization (`SimpleTokenizer`), input sequence formatting (concatenation of states with `[SEP]`, masking for the target state), model configuration (hidden size, layers, heads), and output processing.
    *   **Deliverable:** A working PyTorch implementation of the `ECADynamicsTransformer` class, capable of taking a sequence of past ECA states and predicting the next state.

*   **Task 2.2: ECA Training Data Generation Pipeline:**
    *   **Action:** Adapt or implement functions (referencing `generate_ca_dataset` from Burtsev's code) to generate large-scale training, validation, and test datasets for the O-S task.
    *   **Parameters:**
        *   ECA rule diversity (e.g., all 256 elementary rules, or representative samples from Wolfram classes).
        *   ECA `radius` (likely `r=1` for elementary CAs, but configurable).
        *   ECA `width` (lattice size, e.g., 20, 64, 128 – needs to match expected input from binarized time series).
        *   Number of history steps (`steps` in Burtsev's `CellularAutomataDataset`) fed to the Transformer.
        *   Total evolution time `T` per CA sample.
    *   **Output Format:** Data formatted as `(input_sequence_of_past_states, target_next_state)` suitable for the chosen Transformer architecture (e.g., masked language modeling setup).
    *   **Deliverable:** Scripts to generate and save these datasets (e.g., in Hugging Face `datasets` format or simple JSON/Parquet).

*   **Task 2.3: Training and Validation of `ECADynamicsTransformer`:**
    *   **Action:** Develop a robust training script for the `ECADynamicsTransformer`.
    *   **Key Elements:**
        *   Appropriate loss function (e.g., CrossEntropyLoss on masked tokens if using `BertForMaskedLM`).
        *   Optimizer (e.g., AdamW).
        *   Learning rate schedule.
        *   Regular evaluation on a validation set (metrics: next-state prediction accuracy per cell, per row).
        *   Experiment tracking (e.g., Weights & Biases, as seen in Burtsev's code).
    *   **Goal:** Train a model that generalizes well to predicting evolutions of seen and, ideally, unseen ECA rules.
    *   **Deliverable:** Trained model weights for the `ECADynamicsTransformer` (e.g., `.pth` file) and training/validation logs.

*   **Task 2.4: Implementation of `assess_fit` Method:**
    *   **Action:** Implement the `assess_fit(self, binary_pattern_2d, ...)` method within the `ECADynamicsTransformer` class (or as a utility function using the trained model).
    *   **Functionality:**
        1.  Takes a 2D binary array (e.g., `B_mils_segment`) as input.
        2.  Uses a sliding window approach to create `(history, actual_next_state)` sub-sequences from this input pattern.
        3.  Feeds these `history` sequences to the trained Transformer to get predictions for the `next_state`.
        4.  Compares predictions with `actual_next_state` to calculate an aggregate "fit score" (e.g., average per-cell prediction accuracy, average negative log-likelihood of true next states, or a self-certainty inspired metric).
    *   **Deliverable:** A robust `assess_fit` method that returns a scalar score.

*   **Task 2.5 (Potential Extension/Stretch Goal): Rule Inference Capability:**
    *   **Action (If time permits or for future iterations):** Explore adapting the Transformer to also perform the "Orbit-State+Rule (O-SR)" task, i.e., predict the ECA rule number itself.
    *   This would involve adding a classification head to the Transformer and modifying the training data and loss function.
    *   **Deliverable (If pursued):** Enhanced `ECADynamicsTransformer` with rule prediction capabilities.

**3. Key Tools & Technologies:**

*   Python
*   PyTorch
*   Hugging Face `transformers` library (especially for `BertForMaskedLM` and `BertConfig`)
*   `cellpylib` (for ECA data generation)
*   `numpy`, `pandas`
*   Hugging Face `datasets` (for managing training data)
*   `tqdm` (for progress bars)
*   Weights & Biases (for experiment tracking, optional but recommended)

**4. Integration Points & Dependencies:**

*   **Input to `assess_fit`:** Will receive MILS-processed 2D binary arrays from the "Data Encoding & MILS Specialist." Clear interface and data format (e.g., NumPy array dimensions) needed.
*   **Output of `assess_fit`:** Will provide a scalar score used in the "Optimal Binary Encoding Scheme Discovery" experiment, and potentially to augment the final MDL score in the main pipeline.
*   **GA Enhancement:** The trained Transformer might be used by the "ECA Modeling & GA Specialist" to guide or prune the GA search for ECA rules.

**5. Evaluation & Success Criteria for this Component:**

*   **Primary:** The `ECADynamicsTransformer` achieves high next-state prediction accuracy on a held-out test set of ECA evolutions (including potentially unseen rules).
*   **Secondary:** The `assess_fit` method, using the trained Transformer, produces scores that are:
    *   Meaningful (e.g., higher for patterns clearly generated by simple ECAs vs. random patterns).
    *   Discriminative when used in the "Optimal Binary Encoding Scheme Discovery" experiment (i.e., helps select an encoding scheme that leads to good final ROC AUC).
*   **Tertiary (If Rule Inference is pursued):** Reasonable accuracy in predicting the correct ECA rule number.

**6. Timeline & Milestones (Illustrative):**

*   **Milestone 1:** Successful replication/adaptation of Burtsev's O-S Transformer training with generated ECA data.
*   **Milestone 2:** Trained `ECADynamicsTransformer` achieving target validation accuracy on next-state prediction.
*   **Milestone 3:** Initial implementation and testing of the `assess_fit` method.
*   **Milestone 4:** `assess_fit` method integrated into the "Optimal Binary Encoding Scheme Discovery" experiment, providing results for encoding selection.

**7. Potential Challenges:**

*   Training data generation at scale.
*   Hyperparameter tuning for the Transformer.
*   Defining the most effective "fit score" in the `assess_fit` method.
*   Computational resources for training.

