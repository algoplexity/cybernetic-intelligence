
---

# **KGRAG System: The Operational Playbook**

### **1. Your First Task (Do This Today)**

1.  **Sync the SharePoint Folder:** Make sure the "Compliance Analyst Co-Pilot" folder is synced to your machine.
2.  **Create Your Workspace:** Make a personal copy of the `1_Source_Code/` folder as described in the Golden Rule.
3.  **Run the Final Validation:** Open a terminal, `cd` into your personal code folder, and run:
    `python final_query_validation.py`
4.  **Watch it work.** It will automatically find the assets in the `2_Model_Assets/` and `3_Data_And_Reports/` folders and run the analysis.

### **2. How to Add New Data (Next Quarter)**

1.  **Get the New Data:** Download the new register dump `all-statement-information_YYYY-MM-DD.csv`.
2.  **Put it in the Right Place:** Save it to the `3_Data_And_Reports/Input/` subfolder on SharePoint.
3.  **Create a New Workspace:** Make a fresh copy of the `1_Source_Code/` folder.
4.  **Run the Pipelines:** Open your terminal and run the scripts from your new workspace **in order**:
    ```bash
    python 1_ingest_register.py
    python 2_process_downloads.py
    python 3_build_production_kg.py
    python 4_train_production_resumable.py
    ```
5.  **Promote the New Assets:** When the run is complete, copy the new, final model from your workspace's output folder into the main `2_Model_Assets/` folder, renaming the old one as a backup.

### **3. How to Improve It (Your Job Now)**

| Want to... | Do This |
| :--- | :--- |
| **Make the system find *better* sentences?** | 1. In your **personal code folder**, run `prospect_for_annotation.py`. <br> 2. Manually label the output file. <br> 3. Re-run `train_operational_classifier.py` to create a new `Classifier v3`. <br> 4. Test it. If it's better, this new model becomes the official one in `2_Model_Assets/`. |
| **Build a dashboard?** | Load the `final_query_validation.py` script into a **Streamlit** or **Flask** app. The `answer_query_rich` function is your API. Point it to the official model and graph assets on SharePoint. |
