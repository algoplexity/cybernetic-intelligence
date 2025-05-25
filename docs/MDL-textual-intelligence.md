**Project Proposal: Symbolic Compression and Synthetic Data Generation using MDL for Survey Analysis**

---

**Project Title**:
Symbolic Motif Discovery and Synthetic Survey Response Generation via MDL-Based Compression Framework

**Context & Motivation**:
Public survey responses often contain rich qualitative data that reflect diverse community insights. However, many such datasets are limited in volume and may not capture the full variety of potential themes. Critically, some responses include sensitive, emotionally charged, or abusive language—particularly in scenarios involving personal trauma, domestic violence, or substance abuse. This sensitivity presents ethical and legal barriers to sharing data openly or conducting reproducible experiments.

To address this, we propose generating synthetic data that retains the thematic and structural characteristics of real responses while ensuring no exposure of private or harmful content. This enables safe, privacy-preserving research using state-of-the-art open-source libraries on public platforms such as free-tier Google Colab, allowing for transparent, reproducible experimentation as we iterate through potential solution pathways. To enhance downstream natural language processing (NLP) tasks such as summarisation, topic detection, and theme frequency analysis, we propose a method for generating high-quality synthetic survey data using a novel symbolic motif discovery approach grounded in Minimum Description Length (MDL) principles.

**Use Case**:
Using a small number of seed responses from a 36-question public survey, our objective is to:

1. **Identify common themes** per question and across the entire survey.
2. **Quantify frequency** of each theme's occurrence across the responses.
3. **Extract supporting text fragments** that substantiate these themes.

This requires both **abstractive** and **extractive summarisation**, which traditional black-box LLM pipelines struggle to explain or validate. Our symbolic, MDL-driven method aims to offer both interpretability and generative flexibility.

---

**Solution Overview**:
We propose a **symbolic motif discovery and generation framework** powered by the Minimum Description Length (MDL) principle. The key idea is to treat LLM token sequences as symbolic data and use MDL to discover recurring patterns (motifs) that can compress and represent the dataset efficiently. Importantly, this goes beyond frequency-based n-gram mining by adaptively selecting motifs based on information-theoretic gain.

### Core Pipeline:

1. **Tokenization**:

   * Convert each response to a token sequence using a consistent LLM tokenizer (e.g., GPT-2, Gemma-compatible).

2. **Motif Discovery via MDL**:

   * Identify candidate token motifs (not just frequent n-grams) by evaluating their impact on overall description length (DL).
   * Adopt motifs only if their substitution leads to a net compression (ΔDL < 0), accounting for both compressed data and the size of the motif table.
   * This approach allows for semantically rich, multi-token phrases rather than shallow statistical co-occurrences.

3. **Compression & Grammar Formation**:

   * Use discovered motifs to compress all token sequences.
   * Store grammar as a symbolic abstraction of thematic structures in the data.

4. **Synthetic Generation**:

   * Sample or recombine motifs to generate new compressed sequences.
   * Decode them back into natural language using the tokenizer.

5. **Analysis Layer**:

   * Perform thematic summarisation using symbolic motifs (abstractive).
   * Extract supporting text by tracing back motif expansions (extractive).
   * Count motif occurrences across responses and questions for frequency analysis.

---

**Cybernetic Augmentations Addressing Key Shortcomings**:

**Shortcoming 2: Insufficient Nuance and Specificity in Auto-Quantified Themes**

* *Enhancement*: Instead of generic LLM-labeled themes, we leverage human-in-the-loop refinement of motifs into canonical "common points," followed by LLM-assisted synonym/keyphrase expansion. MDL ensures motifs are specific and compact, enabling more meaningful quantification.

**Shortcoming 3: Underutilization of AI in Early Data Shaping (Phase 1)**

* *Enhancement*: Make Phase 1 interactive. Let analysts guide passage extraction and summaries. The symbolic nature of motifs allows real-time feedback and refinement, aligning AI interpretation with human context awareness.

**Shortcoming 4: Lack of Automated Cross-Cutting Thematic Analysis**

* *Enhancement*: Introduce a "Meta-Theme Synthesis" phase (Phase 3.5). Feed human-curated motifs across QIDs into an LLM to infer higher-order systemic themes. MDL motifs help identify cross-question symbolic patterns more robustly than surface-level tokens.

---

**Benefits**:

* **Data-efficient**: Generates diverse synthetic responses from a small seed dataset.
* **Interpretability**: Symbolic motifs act as transparent building blocks for thematic analysis.
* **Adaptability**: Easily extendable to new inputs, enabling dynamic feedback learning.
* **Analytical Rigor**: Supports both qualitative and quantitative NLP analysis.
* **Privacy-Preserving**: Enables open experimentation without exposure to sensitive or abusive real-world data.

---

**Deliverables**:

* Python-based MDL motif discovery and compression toolkit
* Grammar files representing survey motifs
* Synthetic survey response generator
* Summarisation and frequency counting module
* Optional dashboard to explore themes and motifs across questions

---

**Next Steps**:

* Finalise the tokenization scheme and input format (JSONL confirmed)
* Implement MDL-guided motif discovery (in progress)
* Build summarisation and theme extraction module
* Validate against seed data and tune compression threshold
* Extend to synthetic data generation

---

**Conclusion**:
This project bridges symbolic machine learning, information theory, and NLP. By embedding MDL-driven compression into the heart of survey analysis, we unlock new capabilities for interpretability, theme mining, and realistic synthetic data creation from small public datasets. This methodology directly addresses the shortcomings of generalist LLM summarisation by encoding nuance, supporting human-AI co-creation, and enabling thematic synthesis across boundaries. It is highly publishable and extensible to broader domains beyond surveys (e.g., finance, legal, policy feedback).

---


