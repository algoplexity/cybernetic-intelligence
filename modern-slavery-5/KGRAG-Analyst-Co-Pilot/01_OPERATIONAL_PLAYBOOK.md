
---

# **KGRAG System: The Operational Playbook**
This document is your practical, step-by-step guide to running, maintaining, and improving the Analyst Co-Pilot system.

### **1. Your First Task (Do This Today)**
This task validates your environment and shows you the final output in minutes.

1.  **Sync the SharePoint Folders:** Ensure `Code/`, `Models/`, and `Data/` are synced.
2.  **Open the Master Notebook:** Open `Code/master_pipeline.ipynb`.
3.  **Run All Cells:** From the menu, select "Runtime" -> "Run all". This will execute a scaled-down version of the entire pipeline.
4.  **Watch it work.** It will download data, process it, build a graph, train a model, and show you the final query results.

### **2. How to Refresh All Data (Next Quarter)**
This process rebuilds the knowledge graph and retrains the model with the latest data from the register. Expect it to run for several hours.

1.  **Get New Data:** Download the latest `all-statement-information...csv` to the `Data/Input/` subfolder.
2.  **Open the Master Notebook:** Open a personal copy of `Code/master_pipeline.ipynb`.
3.  **Adjust for Full Scale:**
    *   In the **second code cell** (Stage 1), change `SAMPLE_SIZE = 25` to `SAMPLE_SIZE = -1`.
    *   In the **fifth code cell** (Stage 4), increase `NUM_EPOCHS` (e.g., to 50) and `TOTAL_TRIPLETS` (e.g., to 200000) for a more thorough training run.
4.  **Run All Cells.** The script is resumable. If it times out, you can re-run the training cell to pick up where it left off.
5.  **Promote New Assets:** When complete, manually copy the final trained model (`gfm_retriever_final.pth`) from `Models/gfm_retriever_v1/` to a permanent, versioned folder.

### **3. How to Improve the System (Your Job Now)**
This is how you make the system smarter. Always work within your personal copy of the master notebook.

| Want to... | Do This |
| :--- | :--- |
| **Improve the Evidence Filter?** (`Classifier`) | 1. **Run a Prospecting Notebook** (e.g., a copy of `prospect_for_annotation.py`) to find new sentence candidates from the latest data. <br> 2. **Manually label** the output `.jsonl` file for "operational evidence." <br> 3. **Run a Training Notebook** (e.g., `train_operational_classifier.py`) to create a `Classifier v3`. <br> 4. Update the `CLASSIFIER_PATH` in the master notebook to point to your new `v3` model and test its impact. |
| **Improve Entity Extraction?** (`NER`) | 1. **Create a new annotation dataset** focused on the entities you want to improve (e.g., more examples of specific `CONTROL` types). <br> 2. **Run a Training Notebook** (e.g., `train_ner.py`) to create `NER v3`. <br> 3. Update the `NER_PATH` in the master notebook. |

### **4. Emergency Button (If It Breaks)**
Use this if your personal copy of the master notebook is broken.
1.  **Delete it.**
2.  **Make a fresh copy** of the original `Code/master_pipeline.ipynb`.
3.  **Start again.** This is fast and safe; it will not delete your large data or model assets.

---
