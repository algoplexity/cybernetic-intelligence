
---

# **KGRAG System: The Operational Playbook**

This document is your practical, step-by-step guide to running, maintaining, and improving the Analyst Co-Pilot system.

### **1. Your First Task (Do This Today)**

This task will validate that your environment is set up correctly and that the core reasoning engine is working.

1.  **Sync the SharePoint Folder:** Make sure the "Compliance Analyst Co-Pilot" folder is synced to your local machine.
2.  **Create Your Workspace:** Right-click on the `1_Source_Code/` folder and copy it. Rename the copy to `1_Source_Code_MyDevBranch_YYYY-MM-DD/`. You will **only** work inside this personal folder.
3.  **Run the Final Validation:** Open a terminal (like Command Prompt or PowerShell), navigate into your personal code folder, and run the following command:
    `python final_query_validation.py`
4.  **Watch it work.** The script will load the pre-built graph and the pre-trained model from the main asset folders (`2_Model_Assets/` and `3_Data_And_Reports/`) and run two example queries.
5.  **Challenge it:** Open `final_query_validation.py` in a code editor, change the query text in the last two lines, and re-run the script to ask a new question.

### **2. How to Add New Data (Next Quarter)**

This process rebuilds the entire Knowledge Graph and retrains the GNN model with the latest data.

1.  **Get the New Data:** Download the new register dump `all-statement-information_YYYY-MM-DD.csv`.
2.  **Put it in the Right Place:** Save it to the `3_Data_And_Reports/Input/` subfolder.
3.  **Run the Pipelines:** Open your terminal, navigate to your personal code folder, and run the main pipeline scripts **in order**:
    ```bash
    python 1_ingest_register.py
    python 2_process_downloads.py
    python 3_build_production_kg.py
    python 4_train_production_resumable.py
    ```
4.  **Promote the New Assets:** When the run is complete, the new, trained model will be in your run's output folder. Manually copy this final model (e.g., `gfm_retriever_final.pth`) to the main `2_Model_Assets/` folder. It's good practice to rename the old model as a backup (e.g., `gfm_retriever_final_ARCHIVED_YYYY-MM-DD.pth`).

### **3. How to Improve It (Your Job Now)**

This is how you make the system smarter. Always work within your personal `1_Source_Code_.../` copy.

| Want to... | Do This |
| :--- | :--- |
| **Improve the Evidence Filter?** | 1. Run `prospect_for_annotation.py` to get new sentence candidates. <br> 2. Manually label the output `.jsonl` file. <br> 3. Run `train_operational_classifier.py` to create a new `Classifier v3`. <br> 4. Test it. If it's better, copy the new model folder into `2_Model_Assets/`. |
| **Extract New Entity Types?**| 1. Manually create a new annotation dataset. <br> 2. Run `train_ner.py` to create `NER v3`. <br> 3. Update the `build_production_kg.py` script to handle the new entity type. |
| **Build a Dashboard?** | Use a tool like **Streamlit** or **Flask**. The `final_query_validation.py` script contains the core logic. Your app will load the model and call the `answer_query_rich` function to get its results. |

### **4. Emergency Button (If It Breaks)**

*(This is the restored and corrected section.)*

Use this procedure if you've made changes to the code in your personal workspace and the system has stopped working. This will **not** delete any of the large data or model assets.

1.  **Delete your personal code folder.** For example, delete `1_Source_Code_MyDevBranch_YYYY-MM-DD/`.
2.  **Make a fresh copy** of the original, known-good `1_Source_Code/` master folder.
3.  **Start again** from the new copy.

This is your manual "reset to a known good state." Because all the large, slow-to-build assets are stored separately in the `2_Model_Assets/` and `3_Data_And_Reports/` folders, this reset is fast and safe.

---




