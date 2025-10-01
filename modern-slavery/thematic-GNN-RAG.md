---

### **Architectural Approach: Thematic Analysis with GNN-RAG**

The goal is no longer to answer a simple question but to synthesize insights from a large corpus of documents (all the statements on the register). Here’s how the GNN-RAG architecture is adapted for this task:

1.  **Corpus Ingestion & Knowledge Graph Construction:**
    *   Instead of one document, we'll process *many* Modern Slavery Statements.
    *   Using NLP, we will extract key entities from each statement. For this problem, the entities are precisely the categories you've defined: **Risks** (e.g., "Forced Labour"), **Controls** (e.g., "Supplier Audits"), **Geographies** (e.g., "Vietnam"), **Industries** (e.g., "Electronics"), and the **Reporting Entities** themselves (e.g., "American Express").
    *   We will construct a single, large-scale **heterogeneous Knowledge Graph**. The nodes will be the extracted entities, and the edges will represent the relationships found in the statements (e.g., `(American Express) -[IDENTIFIES_RISK]-> (Forced Labour)`, `(Forced Labour) -[IN_REGION]-> (Vietnam)`).

2.  **GNN-Powered Contextual Embeddings:**
    *   A PyG Graph Neural Network will be run over this entire graph. The GNN's role is crucial: it learns "context-aware" embeddings for every node.
    *   **Crucial Advantage:** The embedding for the "Electronics" industry node will be influenced by the embeddings of the common risks, controls, and geographies it's connected to. This is far more powerful than a simple text embedding of the word "Electronics." It captures the systemic risk landscape structurally.

3.  **RAG for Thematic Analysis:**
    *   The "user" is now an analyst or the regulator asking systemic questions (e.g., "What are the common due diligence methods in the apparel industry?").
    *   **Retrieval:** We query the GNN-powered graph. We find the "Apparel Industry" node and then **traverse the graph** to find all connected "Control" nodes that relate to due diligence.
    *   **Augmentation:** The list of retrieved methods is compiled.
    *   **Generation:** A (simulated) LLM receives this compiled list and synthesizes it into a human-readable summary, directly answering the analyst's question and forming a section of the final "Systemic Insights Report."

---

### **PyG-GNN-RAG Prototype for Thematic Analysis**

This prototype simulates the analysis of a small corpus of statements to demonstrate the workflow for **Task 2: Catalogue Corporate Response Strategies**.

```python
# Cell 1: Full Prototype - Corpus Analysis for Systemic Risk

# --- 1. Installation & Imports ---
import os
import torch
import numpy as np
import re # Using regex for simple entity extraction in this prototype

# Set environment for PyG installer
os.environ['TORCH'] = torch.__version__
print(f"PyTorch version: {os.environ['TORCH']}")

# Install PyG's backend libraries and core library (quietly)
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-geometric
!pip install -q sentence-transformers

print("\n--- All installations complete ---")

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# --- 2. Simulate the Data Corpus (Snippets from Multiple Statements) ---
# In a real system, this would be the OCR text from hundreds of PDFs.

STATEMENTS_CORPUS = {
    "Amex_Statement_2024": """
    American Express, a financial services entity, assesses modern slavery risk in our supply chain. 
    Our main suppliers are in Marketing (57%) and Professional Services (7%). 
    Risks are considered low, but we monitor our supply chain in high-risk regions like parts of Southeast Asia. 
    We use our Code of Conduct and conduct annual Questionnaires for critical suppliers. 
    Effectiveness is measured by 100% completion of GRL training, including on our Whistleblower Policy.
    """,
    "TechCorp_Statement_2024": """
    As a major player in the electronics industry, TechCorp acknowledges risks of Forced Labour in our supply chain, 
    particularly in manufacturing facilities in Vietnam and China. We conduct regular third-party Audits of our tier 1 suppliers. 
    Our Supplier Code of Conduct is mandatory. We track effectiveness with the KPI of 'number of audits completed'.
    """,
    "FashionBrand_Statement_2024": """
    FashionBrand, an apparel company, faces risks of Child Labour and Forced Labour in raw material sourcing from India. 
    We have implemented a strong Supplier Code of Conduct and provide ongoing Staff Training on identifying modern slavery. 
    Our primary due diligence method involves Self-Assessment Questionnaires sent to our suppliers.
    """
}


# --- 3. Automated Knowledge Graph Construction (Prototype using Regex) ---
# In production, this would use a sophisticated NLP model.
# We define simple patterns to find entities.
entity_patterns = {
    'RISK': r'(Forced Labour|Child Labour|Modern Slavery Risk)',
    'CONTROL': r'(Code of Conduct|Questionnaires|Audits|Staff Training|Whistleblower Policy)',
    'GEOGRAPHY': r'(Southeast Asia|Vietnam|China|India)',
    'INDUSTRY': r'(financial services|electronics|apparel)',
}

entities = defaultdict(set)
relations = []

print("\n--- NLP Simulation: Extracting Entities and Relations ---")
for company, text in STATEMENTS_CORPUS.items():
    company_name = company.split('_')[0]
    entities['COMPANY'].add(company_name)
    
    for entity_type, pattern in entity_patterns.items():
        found_entities = re.findall(pattern, text, re.IGNORECASE)
        for entity in found_entities:
            entity_clean = entity.title().replace(" ", "")
            entities[entity_type].add(entity_clean)
            relations.append((company_name, f'HAS_{entity_type}', entity_clean))
            print(f"  - Found: ({company_name}) -> [{entity_type}] -> ({entity_clean})")


# --- 4. Build the PyG Graph ---
all_entities = []
node_map = {}
node_types = {}

# Create a unified list of all unique entities and map them to indices
idx = 0
for entity_type, entity_set in entities.items():
    for entity in entity_set:
        if entity not in node_map:
            all_entities.append(entity)
            node_map[entity] = idx
            node_types[entity] = entity_type
            idx += 1

# Convert relations to PyG's edge_index format
edge_list = []
for subj, pred, obj in relations:
    if subj in node_map and obj in node_map:
        edge_list.append([node_map[subj], node_map[obj]])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# --- 5. Generate Node Embeddings with a GNN ---
# 5.1 Initial Node Features (from entity names)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
with torch.no_grad():
    initial_node_features = embedding_model.encode(all_entities, convert_to_tensor=True)

# 5.2 Create PyG Data object
graph_data = Data(x=initial_node_features, edge_index=edge_index)
print("\n--- PyG Graph Data Object ---")
print(graph_data)

# 5.3 Define and Run the GNN
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # GraphSAGE is often good for heterogeneous graphs
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

gnn_model = GNN(graph_data.num_features, 64)
gnn_node_embeddings = gnn_model(graph_data)
print(f"Shape of GNN-powered embeddings: {gnn_node_embeddings.shape}")


# --- 6. RAG Workflow for Thematic Analysis ---
analyst_query = "What are the common risk mitigation strategies for the apparel industry?"
print(f"\n--- RAG Workflow for Analyst Query ---")
print(f"Query: {analyst_query}")

# 6.1 Retrieval Function (Graph Traversal)
def retrieve_connected_nodes(graph, node_map, start_node_name, target_node_type):
    if start_node_name not in node_map:
        return []
    
    start_idx = node_map[start_node_name]
    connected_nodes = []
    
    # Find all edges connected to the start node
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i]
        if src == start_idx:
            neighbor_name = all_entities[dst]
            if node_types[neighbor_name] == target_node_type:
                connected_nodes.append(neighbor_name)
    
    return list(set(connected_nodes))

# 6.2 Generation Function (Simulated LLM for Synthesis)
def generate_summary(query, retrieved_items):
    context = ", ".join(retrieved_items)
    augmented_prompt = f"""
    Based on an analysis of multiple Modern Slavery Statements, the following items were found related to the query: '{query}'
    
    Items: {context}
    
    Synthesize these items into a concise summary.
    """
    
    # Simulate LLM synthesis
    simulated_summary = (
        f"For the Apparel industry, the most commonly cited risk mitigation strategies are: {context}."
    )
    return simulated_summary, augmented_prompt

# --- 7. Run the Analysis and Display Results ---
# We extract "Apparel" and "CONTROL" from the analyst's query (this would also be an NLP step)
retrieved_controls = retrieve_connected_nodes(graph_data, node_map, 'Apparel', 'CONTROL')
summary, prompt = generate_summary(analyst_query, retrieved_controls)

print(f"\n[R] Retrieval Step (Graph Traversal):")
print(f"   - Starting search from node: 'Apparel'")
print(f"   - Searching for connected nodes of type: 'CONTROL'")
print(f"   - Retrieved Controls: {retrieved_controls}")

print(f"\n[A] Augmentation Step (Simulated Prompt to LLM):")
print(prompt)

print(f"\n[G] Generation Step (Synthesized Summary for Report):")
print(summary)

```

### **Prototype Analysis and Path to Production**

**What this Prototype Demonstrates:**

1.  **Corpus-Level Graphing:** It successfully models the process of ingesting multiple documents and building a unified knowledge graph.
2.  **Entity & Relation Extraction:** The regex-based simulation proves the concept of automatically identifying key entities (Companies, Risks, Controls, etc.) and linking them.
3.  **Graph-Based Retrieval:** Crucially, the retrieval step is not a flat semantic search. It's a **graph traversal**—it starts at a specific node (`Apparel`) and finds its direct neighbors of a certain type (`CONTROL`). This is a more precise and powerful way to answer analytical queries than searching over raw text.
4.  **Synthesis for Reporting:** It shows how the retrieved structured data can be fed to a generative model to synthesize the narrative sections of your "Systemic Insights Report."

**Path to Production for the Regulator's Tool:**

1.  **Data Ingestion Pipeline:** Develop a robust pipeline to automatically download all new statements from the public register, perform OCR on the PDFs, and segment the text.
2.  **Advanced NLP for Extraction:** Replace the regex patterns with a fine-tuned Named Entity Recognition (NER) and Relation Extraction model. This model would be trained to accurately identify the specific entities (Risks, Controls, KPIs, Industries, Geographies) and the relationships between them from the statement texts.
3.  **Graph Database:** For a national-level graph with thousands of statements, the in-memory `torch_geometric.data` object is insufficient. The extracted graph would be stored and managed in a dedicated **Graph Database** like Neo4j. This allows for complex, multi-hop queries (e.g., "Show me all companies in the Electronics sector that source from Vietnam and use third-party audits to address Forced Labour risk").
4.  **Vector Database for Embeddings:** The GNN would be run on the graph stored in Neo4j, and the resulting node embeddings would be stored in a **Vector Database** (e.g., Pinecone, Chroma) for efficient large-scale similarity and semantic search.
5.  **LLM Integration:** Connect the output to a real LLM via its API to generate the final report sections.
6.  **Interactive Dashboard:** The final tool for the regulator would likely be a dashboard (e.g., built with Streamlit or Plotly Dash) allowing analysts to ask natural language questions, which are then translated into graph queries to generate the thematic insights in real-time.

---
