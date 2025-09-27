
---

### **Project: "Modern Slavery KG" Micro-PoC**
**Timeline:** 1 Week
**Objective:** To validate that we can create a high-quality, queryable Knowledge Graph (KG) from a real-world Modern Slavery Statement, using a secure, two-phase (Extraction/Synthesis) process.

#### **Phase 0: Setup & Ontology Definition (Day 1-2)**

This foundational phase ensures we have the right tools and a clear, legally-grounded schema for our graph.

1.  **Environment Setup:**
    *   Establish a local Python environment.
    *   Install the necessary libraries (`kg-gen`, `dspy`, etc.).
    *   Set up and run a local LLM instance via **Ollama**. We will start with a fast, capable model like `Llama3` for the initial extraction.

2.  **Ontology Finalization:**
    *   We will formalize the ontology derived from **Section 16 of the Modern Slavery Act**. This will be our `context` for the `kg-gen` tool.
    *   **Final Ontology (Entity & Relation Types):**
        *   **Entities:** `Reporting Entity`, `Governance Body`, `Industry Sector`, `Geographic Location`, `Supplier Category`, `Identified Risk`, `Control Measure`, `Due Diligence Process`, `Effectiveness Metric`, `Consultation Process`.
        *   **Relations:** `has_governance_body`, `operates_in_sector`, `operates_in_location`, `has_supplier_category`, `identifies_risk`, `uses_control_measure`, `implements_due_diligence`, `measures_effectiveness_with`, `consults_with`, `is_located_in`, `is_linked_to`.

3.  **Input Data Preparation:**
    *   The 9-page **American Express Modern Slavery Statement 2024** will be converted to a clean text file (`amex_statement.txt`).

#### **Phase 1: Knowledge Graph Extraction (Day 3)**

This phase simulates the secure, "encoder-only" process of transforming the unstructured document into a structured asset.

1.  **Execution:** We will run the `kg-gen` pipeline on the prepared text file.
    *   The core command will be a Python script that calls the library's `generate` and `cluster` functions.
    *   **Parameters:**
        *   `document_text`: Content of `amex_statement.txt`.
        *   `context`: The finalized ontology from Phase 0.
        *   `llm`: The local Ollama `Llama3` model.
        *   `chunk_size`: 4000.
        *   `cluster`: `True`.

2.  **Output (The Air-Gapped Asset):**
    *   `amex_clustered_graph.json`: The structured knowledge graph.
    *   `amex_clustered_graph.html`: The interactive visualization of the graph.

3.  **Initial Validation:** A brief (~1 hour) sanity check of the visualization to ensure the clusters and relationships are logically sound.

#### **Phase 2: Analysis & Synthesis (Day 4-5)**

This phase simulates the "decoder-only" analysis, proving the value of the KG and the viability of the air-gap. **All work in this phase will be done using *only* the `.json` file from Phase 1.**

1.  **Define Test Queries:** We will test the KG's utility by answering the following key compliance questions:
    *   **Q1 (Governance):** "Summarize the governance structure American Express has in place for modern slavery."
    *   **Q2 (Risk & Due Diligence):** "What are the primary `Control Measures` and `Due Diligence Processes` used by American Express?"
    *   **Q3 (Effectiveness):** "How does American Express measure the effectiveness of its actions?"
    *   **Q4 (Complex Query):** "Show the relationship between the `Board` (as a `Governance Body`), the `Identified Risks`, and the `Effectiveness Metrics`."

2.  **Simulate `KG2RAG` Retrieval:**
    *   For each query, we will write a small Python script that:
        *   Loads `amex_clustered_graph.json`.
        *   Performs a keyword search to find the initial relevant nodes (e.g., search for "governance" for Q1).
        *   Performs a 1-hop graph traversal from those initial nodes to "expand" the context, gathering all directly connected entities and relationships.

3.  **Execute Synthesis:**
    *   The retrieved subgraph (a small JSON object) will be formatted into a prompt.
    *   This prompt will be sent to a powerful reasoning model via Ollama (e.g., `Mistral` or a larger Llama3 variant) to generate the final, human-readable answer.
    *   **Prompt Template:**
        > "You are a compliance analyst. Based *only* on the following structured data from a company's Knowledge Graph, provide a comprehensive answer to the user's question.
        >
        > **Knowledge Graph Data:**
        > ```json
        > [Retrieved subgraph data from step 2]
        > ```
        > **User Question:**
        > `[Test query, e.g., "How does American Express measure the effectiveness of its actions?"]`"

#### **Phase 3: Final Validation & Review (End of Day 5)**

We will compare the LLM-generated answers from Phase 2 against a manual reading of the original Amex statement to score the prototype's success.

**Success Criteria:**
*   [ ] **(Extraction Quality):** The clustered graph from Phase 1 is coherent and accurately reflects the key entities in the Amex statement.
*   [ ] **(Air-Gap Viability):** The graph contains sufficient information to answer at least 3 out of the 4 test queries accurately without referencing the source text.
*   [ ] **(Synthesis Quality):** The final generated answers are factually correct, comprehensive, and logically derived from the provided subgraph data.

If we can check all three boxes, this Micro-PoC will be a resounding success, and we will have a powerful, evidence-based case for scaling this approach to the entire Modern Slavery Register.

Let's begin.
