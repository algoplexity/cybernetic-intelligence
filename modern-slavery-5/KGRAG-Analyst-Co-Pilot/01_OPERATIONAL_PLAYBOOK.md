
---

# **KGRAG System: The Operational Playbook**
**Version:** 1.0
**Date:** 6 Nov 2025

This document is your practical, step-by-step guide to running, maintaining, and improving the Analyst Co-Pilot system.

### **1. Your First Task (Do This Today)**
*This task validates your entire environment and the full pipeline in a single, manageable run.*

1.  **Unzip the Handover Package:** Unzip `KGRAG_Analyst_CoPilot_LEAN_Handover...zip`.
2.  **Upload to Drive:** Upload the three resulting folders (`Code`, `Models`, `Data`) to your Google Drive, inside your main project folder.
3.  **Open the Master Notebook:** In your Google Drive, navigate to `Code/` and open `master_pipeline.ipynb` in Google Colab.
4.  **Run All Cells:** From the Colab menu, select **"Runtime" -> "Run all"**.
5.  **Watch it work:**
    *   The first time, it will download several GB of open-source base models. This is a one-time setup.
    *   It will then execute all 5 stages of the pipeline on a small sample of data.
    *   The process will take approximately 1-2 hours.
    *   At the very end, you will see the final query results, confirming the system works.

### **2. How to Run at Full Scale (Quarterly Refresh)**
*This process re-builds the entire knowledge base with the latest data from the register. It is computationally intensive and will take several hours.*

1.  **Get New Data:** Download the latest register dump CSV and place it in the `Data/Input/` folder.
2.  **Make a Workspace Copy:** In your Drive, make a copy of `Code/master_pipeline.ipynb` and rename it (e.g., `master_pipeline_Q4_2025_Run.ipynb`).
3.  **Adjust for Full Scale:** Open your new notebook and make these two changes in the first code cell:
    *   Set `SAMPLE_SIZE = -1` to process all statements.
4.  **Run All Cells:** From the menu, select "Runtime" -> "Run all".
    *   **This will take many hours.** The script will save checkpoints after each training epoch. If your session disconnects, you can simply re-open the notebook and re-run the training cell (`Stage 4`) to resume automatically.
5.  **(Optional) Clean Up:** After the final model (`gfm_retriever_final.pth`) is saved, you can safely delete the large `training_checkpoint.pth` file from the `Models/gfm_retriever_v1/` directory to save space.

### **3. How to Improve the System (Your Job Now)**
*This is how you make the system smarter. Always work within a personal copy of the master notebook.*

| Want to... | Do This |
| :--- | :--- |
| **Improve the Evidence Filter?** (`Classifier`) | **This requires a separate, offline process.** The original scripts for this (`prospect_for_annotation.py`, `train_operational_classifier.py`) should be used to create a new `risk_classifier_model_v3/` folder. Once complete, you would update the `CLASSIFIER_PATH` in the master notebook. |
| **Improve Entity Extraction?** (`NER`) | **This also requires a separate process.** Use the `train_ner.py` script to create a new `ner_model_v3/`. Then, update the `NER_PATH` in the master notebook. |
| **Build a Dashboard?** | 1. Create a new Python script for your dashboard (e.g., `app.py`) using **Streamlit** or **Flask**. <br> 2. Copy the `GNNRetriever` class definition and the `answer_query_rich` function from the master notebook into your app. <br> 3. Your app will load the final assets (`graph.pt`, `gfm_retriever_final.pth`, `metadata.json`) from the `Data/` and `Models/` folders and use the query function to serve results. |

### **4. Emergency Button (If Your Notebook Breaks)**
*Use this if you've made changes to your personal copy of the master notebook and it has stopped working.*

1.  **Delete your broken notebook copy.**
2.  **Make a fresh copy** of the original, known-good `Code/master_pipeline.ipynb`.
3.  **Start again.**

This process is fast and safe. It will not delete the large data or model assets that take hours to generate.
---
