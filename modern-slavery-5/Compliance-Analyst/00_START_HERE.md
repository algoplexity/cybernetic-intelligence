# Zero-Knowledge Developer Handover: Modern Slavery Act Compliance Engine

**"You've never seen this project. In 30 minutes, you'll own it."**
**Date:** 4 Nov 2025 | **Folder:** `ModernSlaveryProject5`

---

### 1. The 3 "Golden Files" That Run Everything
*These 3 files are your new database. They are the only source of truth. Everything else is built from them.*

| File | What It Is |
| :--- | :--- |
| `entity_profiles_v3.parquet` | Every entity in Australia (~12.9M) with its best available name (Legal or Main). |
| `corporate_obligation_log.parquet` | Who **must file** (>$100M/200M revenue), sourced directly from the ATO. |
| `action_log_final_v2.parquet` | Who **actually filed**, correctly "exploded" for every ABN and every financial year. |

---

### 2. The 2 Final Reports You Will Use
*These are the primary outputs for all analysis.*

| File | What It Is |
| :--- | :--- |
| `primary_analytical_report.csv` | The strict, high-certainty, **auditable report**. Use this to find confirmed Non-Lodgers. |
| `multi_pass_investigative_report.csv` | The sophisticated **investigative report**. Use this to find "false negatives" from the primary report. |

---

### 3. The Golden Rules (Never Break These)
*These are our hard-won lessons. They are not optional.*

1.  **ABN is the only high-confidence key.** All strict joins use the ABN. Name and ACN are for low-confidence investigation only.
2.  **A single statement can cover multiple years.** The "explosion" of a statement's `PeriodStart`/`End` into multiple `ReportingYear` records is non-negotiable for accuracy.
3.  **The Primary Report will have "false negatives."** This is by design. Its "Non-Lodger" list is an input for the Investigative Report, which finds the true compliance picture.
4.  **The Source of Truth is Sacred.** Obligations from ATO. Identity from ABR. Actions from the Public Register. Never mix them.
5.  **Use "Published" / "Non-Lodger", never "Compliant".** We report on the action, not the legal quality of the statement.

---

### 4. Your First Task (Do This Today)
*This will prove the entire system works in under 10 minutes.*

1.  **Open a terminal** in the `ModernSlaveryProject5` folder.
2.  **Run the master script:** `python rebuild.py`
3.  **Open the primary report:** `primary_analytical_report.csv`.
4.  **Find a "False Negative":** Filter for **Allianz Australia Limited** (`ABN 21000006226`) and `ReportingYear` **2022-23**. You will see its status is **`Mandated Non-Lodger`**.
5.  **Find the Truth:** Now, open the `multi_pass_investigative_report.csv`. Find the same entity for the same year. You will see its status is **`Published (Name Match)`**.
6.  **You have just validated the entire project.** The system correctly identified a high-certainty non-lodger and then proved it was a false negative through a lower-confidence, investigative process.
