
---
# **Project KGRAG: Charter & Final Report**
**Vision:** To build an intelligent Analyst Co-Pilot that moves beyond keyword search to provide deep, contextual, and systemic insights into modern slavery compliance.

### **1. The Final Architecture: An Ontology-Powered RAG System**
Our final system is a state-of-the-art pipeline that transforms unstructured documents into a queryable knowledge asset. The `master_pipeline.ipynb` executes this process:
*   **Knowledge Graph Construction:** A series of models (`Classifier`, `NER`) read raw text and build a rich, heterogeneous knowledge graph.
*   **GNN Training:** A powerful `GNNRetriever` model is trained on this graph using a contrastive loss function to learn a sophisticated ranking function.
*   **Query Engine:** The trained GNN provides a "learning-to-rank" capability that allows analysts to ask complex conceptual questions and receive highly relevant, evidence-backed answers.

### **2. The R&D Journey: Key Learnings & "Twists and Turns"**
The final architecture was discovered through a rigorous process of prototyping and debugging. The key learnings were:

*   **Foundational Debt and the AIMS Insight:** Our project stands on the shoulders of the **Project AIMS (AI against Modern Slavery)** repository. We **bootstrapped our ontology** and initial models by analyzing their curated, compliance-focused dataset. This was a massive accelerator. However, it also introduced a foundational bias.
*   **The "Classifier as Gatekeeper" Insight:** Our most critical learning was realizing our `v1` classifier, trained on AIMS data, had become an expert in "legal compliance language," not "operational evidence." This biased our entire pipeline, causing it to prefer boilerplate. The breakthrough came when we used our own pipeline to create a new, human-in-the-loop dataset from the Register and trained `Classifier v2`, fundamentally shifting the system's worldview.
*   **The Challenge of the Knowledge Landscape:** We learned that simply building a KG is not enough. The distribution of information matters. Early prototypes struggled because the graph was dominated by generic sentences, causing our GNN to produce generic results. This proved that a high-quality "gatekeeper" (the classifier) is essential to ensure the KG **captures the entire compliance landscape** fairly.
*   **The "Hard Negative Mining" Necessity:** For the GNN to learn to provide expert answers to queries, it needed to be trained on expert-level questions. We discovered that simple contrastive loss was not enough. Implementing **Hard Negative Mining** was the final key to teaching the GNN the subtle difference between a "risk" and a "control," enabling it to correctly answer different questions with different, relevant evidence.

### **3. The Long-Term Vision: Towards a Cybernetic System**
Our current system is a powerful but static "snapshot." The ultimate vision is to evolve this into a self-learning, self-healing system.
*   **The SuperARC Paradigm:** The theoretical foundation for this is Algorithmic Information Theory. The goal is to build a system that doesn't just memorize the knowledge graph but seeks the simplest possible **generative model** to *explain* it.
*   **The Path Forward:**
    1.  **Sense:** Our existing Knowledge Graph pipeline is the "sensory organ." It observes the world.
    2.  **Compare:** The next step is to build an automated "critique" module (likely using a powerful LLM) that can assess the quality and relevance of its own answers.
    3.  **Act:** This critique would then trigger a **model refinement** process, allowing the system to learn and adapt continuously as new information arrives.

This project has successfully built the "sensory organ" and the first "reasoning engine." The next great challenge is to build the automated feedback loop that will bring it to life.
---
