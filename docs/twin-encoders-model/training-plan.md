
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
