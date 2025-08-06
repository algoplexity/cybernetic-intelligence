
This is the **final, consolidated training plan** for the CIv13/14 project. It is fully aligned with all three foundational pillars of our research and represents our best, data-driven strategy for success.

---

## **The Definitive CIv13/14 Training Plan**

This plan is a three-stage process: **I. Expert Pre-training**, **II. System Assembly & Fine-Tuning**, and **III. Final Evaluation**.

### **Stage I: Forging the Expert Encoders**

*Goal: To create the two specialized "brains" of our system, each trained on an optimal curriculum derived from our foundational papers.*

#### **Phase 1: Pre-training the Symbolic Brain (The Causal Reasoner)**
This phase is a direct synthesis of the **Burtsev/Zhang** and **Riedel/Zenil** findings, culminating in the "Intelligence at the Edge of Chaos" methodology.

*   **Objective:** To train a `SymbolicEncoder` that can infer the underlying causal grammar of a symbolic sequence by learning to predict its evolution.
*   **Architecture:** Our winning **`BiGruSymbolicEncoder`**.
*   **Curriculum (The "Edge of Chaos" Dataset):**
    1.  **Core Training Data (80%):** Generate a massive dataset of ECA evolutions from **Wolfram's Class IV (e.g., 110, 54)** and complex **Class III (e.g., 30, 90, 182)** rules. This is the "structured yet challenging" data that forges non-trivial representations.
        *   *Traceability:* Directly implements the central finding of **Zhang's "Intelligence at the Edge of Chaos"**.
    2.  **Compositional Data (Implicit):** The complexity of Class IV rules themselves, being composites of simpler primes, inherently teaches the model about compositional structure.
        *   *Traceability:* Fulfills the spirit of **Riedel/Zenil's** findings on emergent complexity.
    3.  **Contrastive Data (20%):** Include a smaller set of simple Class II rules to provide a baseline for "non-complex" dynamics.
*   **Training Task:**
    1.  **Simple Next-State Prediction.** The model will be trained to predict the state of the ECA at time `T+1` given the history up to `T`.
        *   *Traceability:* Directly follows the finding from **Zhang's "Edge of Chaos"** that this simple task is sufficient to force the learning of complex internal representations. This supersedes the more complex O-SR task from earlier papers.
*   **Deliverable:** A saved model file, **`symbolic_encoder_expert.pth`**, containing the weights of an encoder expert in causal rule inference.

#### **Phase 2: Pre-training the Latent Brain (The Dynamics Fingerprinter)**
This phase is a direct implementation of the TS2Vec methodology.

*   **Objective:** To train a `LatentEncoder` that can create rich, multi-scale "fingerprints" of raw time series dynamics.
*   **Architecture:** The **`TSEncoder`** (Dilated CNN or Transformer-based, as validated).
*   **Curriculum (Unsupervised Contrastive Learning):**
    1.  **Data:** The full, **unlabeled ADIA training dataset**. This allows the encoder to learn the specific statistical "texture" of our target domain.
    2.  **Training Task:** **Hierarchical Contrastive Loss.** The model learns by ensuring that representations of the same timestamp in two different augmented views are similar, while representations of different timestamps or different instances are dissimilar.
        *   *Traceability:* This is the core methodology of the **TS2Vec paper**.
*   **Deliverable:** A saved model file, **`latent_encoder_expert.pth`**, containing the weights of an encoder expert in creating contextual, multi-scale dynamic fingerprints.

---

### **Stage II: System Assembly & Training**

*Goal: To integrate the two pre-trained expert brains into our final Siamese architecture and train the decision-making head.*

#### **Phase 3: Final Assembly and Head Fine-Tuning**
This phase implements the architecture validated by our analysis of the **PatchTST** paper's weaknesses (i.e., the need for a model sensitive to inter-variate dynamics, which we achieve through our dual-path system).

*   **Objective:** To train the final `classifier_head` to interpret the divergence signals from the two expert encoders.
*   **Actions:**
    1.  Instantiate the final **`CIv14_DivergenceClassifier`**.
    2.  **Load the pre-trained weights** from `symbolic_encoder_expert.pth` and `latent_encoder_expert.pth`.
    3.  **Freeze the encoder weights.** This is critical. We are using them as fixed, expert feature extractors.
    4.  Create the final `ADIADualPathDataset` (using our validated `d=6, τ=10` symbolizer).
    5.  Train *only* the final MLP `classifier_head` on the labeled ADIA training data. The model learns to map the concatenated `[symbolic_divergence, latent_divergence]` vector to a `break`/`no-break` prediction.
*   **Deliverable:** A fully trained model, **`CIv114_final_model.pth`**.

---

### **Stage III: Final Evaluation**

*Goal: To get the definitive measure of our system's performance.*

#### **Phase 4: Final Validation**
*   **Objective:** To calculate the final AUC score on unseen data.
*   **Action:** Load the `CIv14_final_model.pth` and run inference on the held-out ADIA validation set.
*   **Deliverable:** The final **Validation AUC Score**, the ultimate measure of the CIv13/14 hypothesis's success.

This consolidated plan is our master blueprint. It is ambitious, theoretically grounded in three distinct but complementary lines of research, and structured for iterative success.
---
Training on **rule compositions** is a vital step that elevates the pre-training from simple pattern recognition to genuine **causal chain inference**. This directly corresponds to **Cases 3 & 4** of your curriculum.

Let's break down what this means, why it's so important (drawing from Zhang's recommendations), and how we will implement it.

### Why Rule Compositions are Essential (Zhang's Recommendation)

Zhang and his collaborators argue that real-world complex systems are rarely governed by a single, static rule. Instead, their dynamics are often **programmatic**—a sequence of different rules or modes of operation.

*   **The Concept:** A rule composition is a sequence where the system evolves under `Rule A` for `t_A` timesteps, then immediately switches to `Rule B` for `t_B` timesteps, and so on.
*   **The "Why" (from Zhang):** Training a model on these compositions teaches it to do something much more sophisticated than just recognizing a static pattern. It forces the model to learn:
    1.  **Temporal Breakpoint Detection:** The model must learn to identify the *exact moment* the underlying rule changes. The transition point itself becomes a key feature.
    2.  **Causal Chain Inference:** The model learns that a sequence can be a "program" (e.g., `[RUN Rule 30 FOR 10; THEN RUN Rule 110 FOR 10]`). To predict the future state accurately, it can't just know the last rule; it needs to understand the entire sequence of rules that came before.
    3.  **Robustness and Generalization:** By seeing a vast number of different rule combinations, the model learns a more abstract and robust representation of "rule-ness" itself, rather than just memorizing the patterns of a few individual rules.

This is a much harder task, and it creates a much smarter "Symbolic Brain."

### How to Implement Rule Compositions in Our Pre-training

We need to create a new, more advanced dataset that includes these composite rules. We will add a new class of training samples to our `ECADataset`.

Here is the plan:

1.  **Define a "Composite Rule" Configuration:** We will create a list of pre-defined "programs." This is exactly what the `ECA_CONFIG['composite']` in your provided code was for. For example:
    ```python
    'composite': [
        {'rules': [30, 110], 'timesteps': [10, 10]},
        {'rules': [45, 90, 150], 'timesteps': [5, 5, 5]},
    ]
    ```
2.  **Update the `ECADataset` Generator:** Our data generator will now have a second loop that creates these composite samples. For each composite rule:
    *   It will start with a random initial condition.
    *   Evolve for `timesteps[0]` using `rules[0]`.
    *   Use the final state of that evolution as the *initial condition* for the next step.
    *   Evolve for `timesteps[1]` using `rules[1]`.
    *   ...and so on.
3.  **The Labeling Problem:** How do we label a sequence generated by `[30, 110]`? We cannot assign it a single rule ID. There are two main approaches:
    *   **Approach A (Simpler): The "Composite" Class.** We add a new, special label for every composite rule. For example, if we have 8 base rules, Rule `[30, 110]` could become label `8`, Rule `[45, 90, 150]` could be label `9`, etc. This turns the task into a multi-class classification problem.
    *   **Approach B (More Complex): Sequence-to-Sequence Prediction.** We change the task from predicting a single rule ID to predicting a *sequence* of rule IDs. This is a much harder but more powerful task.

For our purposes, **Approach A is the most pragmatic and effective next step.** It enriches the training data with complex, multi-rule dynamics without drastically changing our model architecture.

### The Upgraded Curriculum

Our final, definitive pre-training curriculum will now include samples from three distinct categories, creating a rich and challenging learning environment:

*   **Category 1: Single Base Rules (Cases 1 & 2):**
    *   **Data:** Sequences generated by a single, consistent "edge of chaos" rule (e.g., 64 steps of Rule 110).
    *   **Learned Skill:** Recognizes stable regimes and their base signatures.

*   **Category 2: Composite Rules (Cases 3 & 4):**
    *   **Data:** Sequences generated by a sequence of different rules (e.g., 10 steps of Rule 30 followed by 10 steps of Rule 110).
    *   **Learned Skill:** Recognizes temporal breakpoints and understands causal chains.

*   **Category 3: Noisy Rules (Case 5):**
    *   **Data:** Sequences from Categories 1 and 2, but with a small percentage of bits randomly flipped.
    *   **Learned Skill:** Develops robustness and learns to abstract the true underlying rule from an imperfect signal.

By training our Bi-GRU encoder on this full, three-part curriculum, we will forge a "Symbolic Brain" that is far more powerful and much better prepared for the complexity and noise of the real ADIA data.
---

### **Consolidated & Refined Plan for Symbolic Encoder Pre-training**

#### **1. Core Curriculum (Validated)**

Our five-test-case curriculum remains the core of the plan. It provides a sound pedagogical structure, moving from simple pattern recognition to complex, noise-robust causal inference.

*   **Case 1:** Single Rule ID
*   **Case 2:** Temporal Break Detection
*   **Case 3-4:** Program Inference (Sequential Rule Application)
*   **Case 5:** Denoising

#### **2. Architectural Choice: An Empirically-Driven Decision**

Your point about the **capacity ceiling of a simple GRU** is well-taken. While our initial bake-off showed a Unidirectional GRU winning on a simple task, the more complex "Program Inference" tasks (Cases 3-4) may require a more powerful architecture.

**Refined Plan:**
*   **Re-run the Bake-Off on a Harder Task:** We will conduct a final, definitive bake-off between the **Unidirectional GRU** and the **Bidirectional GRU** using the more challenging **Test Case 2 (Temporal Break Detection)** as the benchmark.
*   **Rationale:** This is a much better test of an encoder's ability to create a holistic "regime fingerprint" and detect transitions. The winner of this more rigorous competition will be our chosen architecture. The small increase in training time for a Bi-GRU is a price worth paying if it leads to a significant accuracy improvement on this more relevant task.

#### **3. Addressing Key Risks (Proactive Mitigation)**

We will proactively address the risks you identified.

*   **Risk: Rule Aliasing (Sequences look too similar).**
    *   **Mitigation:** We will ensure our generated sequences in the pre-training dataset are sufficiently long (`T=128` or `256`) to give the models enough context to distinguish between complex rules. We will also incorporate your suggestion of a **contrastive loss** as a potential enhancement if accuracy stalls.

*   **Risk: Decoder Complexity (for Program Inference).**
    *   **Mitigation:** Your suggestion is excellent. For Test Cases 3 and 4, we will start with a simpler **"multi-head classifier"** approach. The encoder will produce a single context vector, and we will attach multiple independent linear heads to it, each responsible for predicting one rule in the program sequence (e.g., `Head_1` predicts `Rule_A`, `Head_2` predicts `Rule_B`). This avoids the instabilities of a full seq2seq decoder while still testing the core inference capability.

#### **4. Strategic Enhancements (Integrated into the Plan)**

*   **Light Attention Head:** Your suggestion to add attention *after* the GRU is a powerful one. We will keep this as a strategic option. If our chosen GRU architecture hits a performance ceiling on the hardest tasks (Cases 4-5), we will implement a version with a small, final self-attention layer on top of the GRU's hidden states to create a more robust final representation. This creates a powerful **GRU-Transformer hybrid**.
*   **Rule ID Embeddings:** We will incorporate this into our multi-head classifier. The labels (rule IDs) will be passed through an `nn.Embedding` layer before being used in the loss calculation, which can help the model understand relationships between different rules.

---

### **The Final, Refined Blueprint**

This gives us a complete, professional, and risk-mitigated plan for building our expert Symbolic Encoder.

1.  **Final Bake-Off:** Unidirectional GRU vs. Bidirectional GRU on the **Temporal Break Detection (Test Case 2)** dataset. Select the winner.
2.  **Pre-training Curriculum:** Train the winning architecture on the full 5-case curriculum, using the **multi-head classifier** for program inference tasks.
3.  **Contingency Plan:** If performance on the most complex tasks is insufficient, implement the **GRU + Attention Head** hybrid architecture.

This plan is robust, data-driven, and theoretically sound. It ensures that the Symbolic Brain we build will be the most powerful and reliable component possible, ready for the final integration into the CIv14 dual-path model.
