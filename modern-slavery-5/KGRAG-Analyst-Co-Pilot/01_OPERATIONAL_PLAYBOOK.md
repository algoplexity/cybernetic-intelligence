
---

# **KGRAG System: The Operational Playbook**
This document is your practical, step-by-step guide to running, maintaining, and improving the Analyst Co-Pilot system.

### **1. Your First Task (Do This Today)**
This task validates your environment and shows you the final output in minutes.

1.  **Sync the SharePoint Folder:** Ensure "Compliance Analyst Co-Pilot" is synced to your machine.
2.  **Create Your Workspace:** Copy the `1_Source_Code/` folder and rename it `1_Source_Code_MyDevBranch/`. You will only work in this personal copy.
3.  **Run the Final Validation:** Open a terminal, navigate into your personal code folder, and run:
    `python 5_validate_model.ipynb`
4.  **Watch it work.** The script loads the pre-built assets and runs two example queries.
5.  **Challenge it:** Open the notebook, change the query text in the final cell, and re-run it.

### **2. How to Refresh All Data (Next Quarter)**
This process rebuilds the knowledge graph and retrains the model with the latest data. Expect it to run for several hours.

1.  **Get New Data:** Download the latest `all-statement-information...csv` to the `3_Data_And_Reports/Input/` subfolder.
2.  **Create a New Workspace:** Make a fresh copy of `1_Source_Code/`.
3.  **Run the Master Pipeline:** Open your terminal, navigate to your new workspace, and run the five notebooks **in order**, one after the other.
4.  **Promote New Assets:** When complete, manually copy the final trained model (`gfm_retriever_final.pth`) from your run's output folder to the main `2_Model_Assets/` folder. Archive the old one.

### **3. How to Improve the System (Your Job Now)**
This is how you make the system smarter. Always work within your personal code folder.

| Want to... | Do This |
| :--- | :--- |
| **Improve the Evidence Filter?** (`Classifier`) | 1. Use `prospect_for_annotation.py` (not in the master pipeline) to find new sentence candidates. <br> 2. Manually label the output file for "operational evidence." <br> 3. Use `train_operational_classifier.py` to create a `Classifier v3`. |
| **Improve Entity Extraction?** (`NER`) | 1. Create a new, focused annotation dataset. <br> 2. Run `train_ner.py` to create `NER v3`. |
| **Improve the Ranking Brain?** (`GNN`) | 1. Open `4_train_model_resumable.ipynb`. <br> 2. Adjust training parameters (e.g., `NUM_EPOCHS`, `TOTAL_TRIPLETS_TO_GENERATE`). <br> 3. Re-run to create a new GNN model. |

### **4. Emergency Button (If It Breaks)**
Use this if you've made changes to your personal code copy and it stops working.

1.  **Delete your personal code folder** (e.g., `1_Source_Code_MyDevBranch/`).
2.  **Make a fresh copy** of the original `1_Source_Code/` master folder.
3.  **Start again.** This is fast and safe; it will not delete your large data or model assets.

---



*(This is the new artifact that captures our journey and strategic vision.)*

