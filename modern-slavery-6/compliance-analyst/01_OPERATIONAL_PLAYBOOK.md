
---

# Modern Slavery Act Compliance Engine: Operational Playbook

**Audience:** The developer or analyst responsible for running and maintaining the engine on a regular basis.

**Working Directory:** `ModernSlaveryProject6`

---

## ⚠️ BEFORE YOU RUN ANYTHING: READ THESE CORE PRINCIPLES ⚠️

This engine is the result of a "Ground-Zero Rebuild" that uncovered numerous, complex flaws in the source data. It is built on a series of hard-won lessons. Understanding and respecting these principles is not optional. Violating them **will** break the defensibility of the final reports and repeat the failures of the past.

### **Principle 1: The Goal is a *DEFENSIBLE* Report, Not a *Complete* One.**
Our primary job is to produce a report with the highest possible level of certainty. This means we **intentionally and correctly quarantine** data that is flawed, ambiguous, or fails our quality checks. An exception log is a **sign of a successful, rigorous process**, not a failure.

### **Principle 2: The Three Universes are Sacred and Separate.**
Each of our three "Golden Assets" is built from a different, authoritative source. We must respect this separation of concerns.
-   **Identity (ABR):** The **only** source of truth for an entity's name and type is `entity_profiles_v3.parquet`.
-   **Obligations (ATO):** The **only** source of truth for an entity's income and mandate is `corporate_obligation_log.parquet`.
-   **Actions (Register):** The **only** source of truth for what an entity did is `action_log_final_v2.parquet`.
**CRITICAL:** *Never* use data from one universe to "correct" another. The name in the ATO file is **not** an authoritative name; it is only a search term.

### **Principle 3: All High-Confidence Matching is Done on `ABN`.**
The ABN is the only reliable key. The **Primary Report** uses strict `ABN`-to-`ABN` joins. The final **Investigative Report** uses a sophisticated waterfall logic (`ABN` -> `Name` -> `ACN`), and the method of the match is always annotated to show the level of confidence.

---

## **The Master End-to-End Rebuild Notebook**

This notebook is the single, executable source of truth for the entire data pipeline. It is designed to be run from top to bottom in a clean `ModernSlaveryProject6` environment to generate all golden assets and final reports from the three raw source files.

---
### **Cell 1: Project Header & Setup**
**Purpose:** To install all required libraries and define the file paths for our clean `ModernSlaveryProject6` working directory. This cell ensures the entire environment is correctly configured before any processing begins.

```python
# ==============================================================================
# @title Cell 1: Project Header & Setup
# ==============================================================================
# --- Imports and Library Installation ---
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import pyarrow.parquet as pq
import pyarrow as pa
import re

# --- Install Fuzzywuzzy (for the investigative report) ---
try:
    from fuzzywuzzy import process
    print("fuzzywuzzy is already installed.")
except ImportError:
    print("Installing fuzzywuzzy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy", "python-Levenshtein"])
    from fuzzywuzzy import process
    print("Successfully installed and imported fuzzywuzzy.")

# --- Define Definitive File Paths in the Clean Environment ---
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    project_folder = Path("/content/drive/MyDrive/ModernSlaveryProject6/")
except ImportError:
    project_folder = Path("./ModernSlaveryProject6/")
    project_folder.mkdir(exist_ok=True)

# Raw Inputs
raw_identity_file = project_folder / "abn_bulk_data_all.parquet"
raw_obligations_file = project_folder / "obligation_universe_first_pass.parquet"
raw_actions_file = project_folder / "all-statement-information_2025-10-27.csv"

# Processed Golden Assets
identity_output_file = project_folder / "entity_profiles_v3.parquet"
obligations_output_file = project_folder / "corporate_obligation_log.parquet"
actions_output_file = project_folder / "action_log_final_v2.parquet"
actions_exceptions_file = project_folder / "action_log_exceptions_v2.csv"

# Final Reports
primary_report_output = project_folder / "primary_analytical_report.csv" # Strict, auditable report
investigative_report_output = project_folder / "multi_pass_investigative_report_final.csv" # Comprehensive final report

print(f"\nSetup complete. All paths point to the clean directory: {project_folder}")
```
---
### **Cell 2: Script 1 - Rebuild the Universe of Identity**
**Purpose:** To create the comprehensive 12.9M-record identity list from the raw ABR data.

**Key Learnings Implemented:**
-   **Lesson 4 (Identity is Broader than "Legal"):** This script implements the crucial `LGL` (Legal) then `MN` (Main) name fallback logic. This is the only way to create a comprehensive universe that includes the high-revenue entities from the ATO data.
-   **Lesson 1 (ABN vs. ACN):** The `ACN` is treated as optional, nullable data. The `ABN` is the primary key.
-   **Defensibility:** The `Name_Source` column is added to provide a transparent audit trail for every entity's name. The entire process is memory-safe and enforces a fixed schema to prevent data corruption.

```python
# ==============================================================================
# @title Cell 2: Script 1 - Build Universe of Identity (entity_profiles_v3.parquet)
# ==============================================================================
# (Content of the final, corrected Script 1.6)
```
---
### **Cell 3: Script 2 - Rebuild the Universe of Obligations**
**Purpose:** To clean and canonize the raw ATO CTT data into our golden `corporate_obligation_log.parquet`.

**Key Learnings Implemented:**
-   **Principle 2 (Source of Truth is Sacred):** This script correctly uses the ATO file as the definitive source for `TotalIncome`. It deliberately drops the non-authoritative `Name` column.
-   **Data Quality:** It immediately canonizes the `ABN` from a number to a clean 11-digit string.

```python
# ==============================================================================
# @title Cell 3: Script 2 - Build Universe of Obligations (corporate_obligation_log.parquet)
# ==============================================================================
# (Content of the final, corrected Script 2)
```
---
### **Cell 4: Script 3 - Rebuild the Universe of Actions**
**Purpose:** To transform the raw, messy `all-statement-information...csv` into our clean, "long" format `action_log_final_v2.parquet`.

**WARNING:** This is the most complex transformation in the engine. It is the solution to the "Allianz" problem.

**Key Learnings Implemented:**
-   **Principle 1 (Defensible, not Complete):** This script begins with a rigorous, 4-step QC process that validates ABN presence, date integrity, annual period duration, and uniqueness. All failing records are defensibly quarantined to `action_log_exceptions_v2.csv`.
-   **Lesson 3 (Explosion is Non-Negotiable):** The core of this script is the two-stage "explosion." It first explodes records by `ReportingYear` (solving the multi-year span problem) and then explodes them by `ABN` (solving the multi-entity joint statement problem).
-   **Defensibility:** It concludes by mathematically proving that the final row count matches an independently calculated "Control Total," ensuring no data was lost.

```python
# ==============================================================================
# @title Cell 4: Script 3 - Build Universe of Actions (action_log_final_v2.parquet)
# ==============================================================================
# (Content of the final, corrected Script 3.5b)
```
---
### **Cell 5: Script 4 - Generate the Primary Analytical Report**
**Purpose:** To build the strict, high-certainty, auditable report on the obligated cohort ("The Auditor's Report").

**Key Learnings Implemented:**
-   **Principle 3 (High-Confidence Matching):** This script uses a strict `ABN`-to-`ABN` join only. Its "Non-Lodger" list is the definitive starting point for further investigation.
-   **Defensibility:** It applies our final, granular classification logic, creating a rich `Reporting_Status` that is legally precise ("Published," not "Compliant") and includes the specific income threshold applied.

```python
# ==============================================================================
# @title Cell 5: Script 4 - Generate the Primary Analytical Report
# ==============================================================================
# (Content of the final, corrected Script 5.4)
```
---
### **Cell 6: Script 5 - Generate the Multi-Pass Investigative Report**
**Purpose:** To find the "false negatives" from the Primary Report. This is our most complete and accurate view of compliance ("The Detective's Report").

**Key Learnings Implemented:**
-   **Lesson 6 (Two-Report Strategy):** This script executes our final, most sophisticated logic to produce a more complete picture than the Primary Report alone.
-   **Lesson 7 (Method of Match Defines Value):** It executes the hierarchical `ABN -> Name -> ACN` waterfall search against the **raw source data**. The output clearly annotates the method of the match, ensuring the confidence level of each finding is transparent.

```python
# ==============================================================================
# @title Cell 6: Script 5 - Generate the Multi-Pass Investigative Report (CORRECTED)
# ==============================================================================
# --- CRITICAL CONTEXT FOR THIS SCRIPT ---
# (The full, detailed context block we drafted goes here)
#
# (The rest of the final, definitively corrected Multi-Pass script follows)
```
---
### **Cell 7: Script 6 - Final Verification EDA**
**Purpose:** To perform a final EDA on the two main reports, providing a high-level summary and proving the coherency of the entire project's output.

```python
# ==============================================================================
# @title Cell 7: Final Verification (CORRECTED)
# ==============================================================================
# (Content of the final, corrected Script 6)
```

---

## **How to Add New Data (The Quarterly/Annual Update Process)**

The engine is designed for repeatable updates. The process is always: **drop the new raw file, then run the specific rebuild script for that universe.** You do not always need to run the entire notebook.

---

### **Scenario 1: You have a new ABR monthly data dump.**
*(This updates our master list of all company names and details.)*

1.  **File:** `abn_bulk_data_all.parquet`
2.  **Action:**
    *   Place the new version of this file into the `ModernSlaveryProject6` directory, overwriting the old one.
    *   In the Master Notebook, **run Cell 2 (Script 1)** to rebuild the `entity_profiles_v3.parquet` golden asset.

---

### **Scenario 2: You have a new ATO Corporate Tax Transparency file.**
*(This updates our list of entities who are legally obligated to report.)*

1.  **File:** `obligation_universe_first_pass.parquet`
2.  **Action:**
    *   Place the new version of this file into the `ModernSlaveryProject6` directory, overwriting the old one.
    *   In the Master Notebook, **run Cell 3 (Script 2)** to rebuild the `corporate_obligation_log.parquet` golden asset.

---

### **Scenario 3: You have a new data export from the Public Register.**
*(This updates our list of who actually published a statement.)*

1.  **File:** `all-statement-information_2025-10-27.csv` (the filename will change with the date).
2.  **Action:**
    *   Place the new CSV file into the `ModernSlaveryProject6` directory.
    *   Update the `raw_actions_file` path in **Cell 1 (Setup)** to point to this new filename.
    *   In the Master Notebook, **run Cell 4 (Script 3)** to rebuild the `action_log_final_v2.parquet` golden asset and its corresponding exception log.

---

### **Generating the Final Reports**

After you have updated one or more of the golden assets using the steps above, you can then generate the final reports:

1.  **Run Cell 5 (Script 4)** to generate the strict, auditable `primary_analytical_report.csv`.
2.  **Run Cell 6 (Script 5)** to generate the comprehensive `multi_pass_investigative_report_final.csv`.

**Best Practice:** For a full, clean quarterly or annual refresh, the recommended procedure is to run the entire notebook from top to bottom (Cells 1 through 7). This ensures all assets and reports are perfectly synchronized with the latest raw data.

---

## **Emergency Button (If It Breaks)**
If data becomes corrupted or a script fails unexpectedly, the fastest way to recover is to reset to a known good state.

1.  **Delete the `ModernSlaveryProject6` folder.**
2.  **Re-create the `ModernSlaveryProject6` folder.**
3.  **Copy the 3 essential raw source files** back into the clean folder.
4.  **Re-run this Master Notebook.**

*All code should be version-controlled in a Git repository. A `git clone` would be the professional equivalent of this process.*

---



