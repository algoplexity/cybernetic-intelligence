
---

### **The Original 4-Phase Product Roadmap (Powered by BERTopic)**

This roadmap is focused on the value delivered to the analyst at each stage.

### **Phase 1: The Evidence Engine**

**Objective:** To ingest, process, and structure the entire universe of modern slavery statements into a single, searchable repository of high-quality, operational evidence. This is the foundation upon which all analysis is built.

**Key Features Delivered:**

*   **Automated Ingestion:** A fully automated system that fetches new statements as they are published, ensuring the evidence base is always current.
*   **The Operational Filter:** Implements **Key Learning #2** at scale. Every sentence from every document is passed through our `risk_classifier_model_v2`. This crucial step ensures that the core Evidence Engine contains only concrete, operational statements, filtering out millions of sentences of generic legal boilerplate.
*   **Unified Search:** A simple but powerful search interface allowing analysts to perform keyword and semantic searches across the entire evidence base.

**How BERTopic Powers This Phase:**

*   While the full BERTopic model isn't trained yet, this phase is about preparing the high-quality, filtered input *for the BERTopic engine*. The quality of this evidence base directly determines the quality of the topics the model will discover in the next phase. The `sentence-transformer` used by BERTopic provides the embeddings for the "Unified Search" feature.

---

### **Phase 2: Statement X-Ray**

**Objective:** To move beyond simple search and provide a deep, thematic analysis of any *single* document. This feature allows an analyst to instantly understand the core focus and substance of a company's statement.

**Key Features Delivered:**

*   **Thematic Fingerprinting:** An analyst can upload or select any statement, and the system instantly generates its "thematic fingerprint"—a dashboard showing the precise percentage of the document dedicated to `RISK`, `CONTROL`, `GOVERNANCE`, and any other discovered topics. It answers the question: "What is this company *really* talking about?"
*   **Evidence Browser:** The thematic fingerprint is interactive. An analyst can click on a theme (e.g., `CONTROL` - 45%) and instantly see every specific sentence from that document that the model has assigned to that topic, complete with source highlighting.

**How BERTopic Powers This Phase:**

*   This is a direct application of a trained, semi-supervised BERTopic model. When a new statement is processed, its sentences are passed to `BERTopic.transform()`.
*   The model assigns a topic ID to each sentence. The "Thematic Fingerprint" is simply a value-count of these topic assignments.
*   The "Evidence Browser" is a simple filter, retrieving all sentences where `topic_id` matches the one selected by the analyst.

---

### **Phase 3: Sector-Wide Intelligence & Comparative Analytics**

**Objective:** To scale the analysis from a single document to the entire dataset, enabling strategic, comparative insights that are impossible to achieve manually.

**Key Features Delivered:**

*   **Peer Benchmarking:** The Co-Pilot can answer questions like, "Compare the thematic focus of Company A, Company B, and Company C." The system generates a side-by-side visualization of their "Thematic Fingerprints," revealing strategic differences in their compliance programs.
*   **Topic Trend Analysis:** The system can plot the prevalence of key topics over time within a specific sector. It answers questions like, "Is the discussion of 'Geopolitical Risk' increasing in the apparel sector since 2022?"
*   **Concept Explorer:** This is the evolution of search. An analyst can search for an abstract concept (e.g., "supplier grievance mechanisms") and the system, using `BERTopic.find_topics()`, identifies the most relevant topic and retrieves the best examples of that concept from hundreds of different companies.

**How BERTopic Powers This Phase:**

*   This leverages the globally trained BERTopic model. "Peer Benchmarking" and "Trend Analysis" are powerful aggregations built on the topic assignments for thousands of documents.
*   "Concept Explorer" uses the core semantic search and topic-finding capabilities of the framework to provide answers based on conceptual relevance, not just keyword matching.

---

### **Phase 4: The Strategic Co-Pilot & Horizon Scanning**

**Objective:** To transform the Co-Pilot from a reactive analysis tool into a proactive, strategic partner that can identify emerging risks and anomalies automatically.

**Key Features Delivered:**

*   **Outlier & Anomaly Detection:** The Co-Pilot can automatically flag statements that are thematically unusual compared to their peers. For example: "Warning: Company X's new statement is 90% `GOVERNANCE` (policy discussion) and only 2% `CONTROL` (actions). This is a significant deviation from their sector average of 35%."
*   **Emerging Topic Radar:** This is the intelligent "self-healing" maintenance loop. The system automatically clusters the sentences that BERTopic classifies as **Outliers (Topic -1)**. When a new, coherent cluster of outliers emerges, the system flags it for analyst review: "Discovery: A new potential topic related to 'AI in supply chain auditing' has been detected. Would you like to promote this to a formal topic?"

**How BERTopic Powers This Phase:**

*   "Outlier Detection" is achieved by comparing the topic distribution of a single document to the average distribution of its peer group.
*   The "Emerging Topic Radar" is the production implementation of our final prototype's insight. It uses the unsupervised discovery power of BERTopic on the set of "unknowns" (Topic -1) to automatically detect conceptual drift and guide the system's evolution with minimal human effort. This solves the long-term maintainability problem in an intelligent, collaborative way.
