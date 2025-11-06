
---

# Zero-Knowledge Developer Handover: Modern Slavery Act Compliance Engine

*"You've never seen this project. In 30 minutes, you'll own it."*

**Date:** 4 Nov 2025 | **Folder:** `ModernSlaveryProject6`

---

## 1. The Goal: A Defensible Compliance Picture

This engine answers one primary question: "Of the entities legally obligated to report under the Act, who did and who did not?". It does this by generating two key, complementary reports from a set of clean, validated "Golden Asset" files.

---

## 2. The 3 "Golden" Files That Run Everything
*These three Parquet files are the engine's database. They are the only source of truth. The Master Notebook builds them from scratch.*

| File | What It Is |
| :--- | :--- |
| `entity_profiles_v3.parquet` | The **Identity Universe**: Every entity in Australia (12.9M) with its best available name (Legal or Main). |
| `corporate_obligation_log.parquet` | The **Obligations Universe**: Who **must file** (>$100M/200M revenue), sourced from the ATO. |
| `action_log_final_v2.parquet` | The **Actions Universe**: Who **actually filed**, correctly "exploded" into a clean record for every ABN and financial year. |

---

## 3. How to Open Them (The Basics)
*No SQL. No cloud. Just Python + Pandas. This is how you can explore any of the `.parquet` files.*

```python
# 1. You may need to install this once in your environment
# pip install pandas pyarrow

# 2. Copy-paste this into a Python script or notebook to open any file
import pandas as pd
df = pd.read_parquet("ModernSlaveryProject6/entity_profiles_v3.parquet")
print(df.head())
```

---

## 4. The Golden Rules (Never Break These)
*These are our hard-won lessons. Violating them will break the engine. Read the `02_PROJECT_CHARTER_AND_LEARNINGS.md` for the full story behind them.*

1.  **The Goal is a *Defensible* Report.** We quarantine flawed data. An exception log is a sign of success.
2.  **The Three Universes are Sacred.** ABR for names. ATO for income. The Public Register for actions. Never mix them.
3.  **High-Confidence Matching is Done on `ABN` Only.** Name/ACN matching is for investigation and must be flagged.
4.  **An Action Can Span Multiple Years. You *Must* Explode It.** This is the "Allianz" lesson and is non-negotiable.
5.  **`Reporting Status` is Not Legal `Compliance Status`.** We report a factual `Published` status, not a legal judgment of "Compliance."

---

## 5. Your First Task (Do This Today)
*This will run the entire engine from scratch and prove it works. This is the ultimate test.*

1.  **Open the Master Notebook:** In your Colab or Jupyter environment, open the `01_OPERATIONAL_PLAYBOOK.md` file. It contains the complete, end-to-end Python script.
2.  **Run the Notebook:** Execute the entire notebook from top to bottom. It will take a significant time to complete as it rebuilds all 12.9M+ records.
3.  **Open the Final Report:** Once complete, open the `multi_pass_investigative_report_final.csv` file from your `ModernSlaveryProject6` folder.
4.  **Find the "Aha!" Moment:**
    *   Filter the report for the entity name **`ALLIANZ AUSTRALIA LIMITED`** and the `Income year` **`2022-23`**.
    *   Observe the result. You will see that the `ABN_Match` status is **`Non-Lodger`**, but the `Name_Match` status is **`Published (as Reporting Entity)`**.
5.  **You have just validated the entire project.** You have proven that the engine correctly identified a "false negative" that a simpler process would have missed.

---


