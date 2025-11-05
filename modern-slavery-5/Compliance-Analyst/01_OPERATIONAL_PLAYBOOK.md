# Operational Playbook: Modern Slavery Act Compliance Engine

**Purpose:** This document provides the complete technical and procedural guide for running and maintaining the compliance engine.

---

### 1. Definitive Folder Structure
*The project is self-contained within the `ModernSlaveryProject5` directory.*

- **`/` (Root Folder: `ModernSlaveryProject5`)**
  - Contains `rebuild.py`, the single, master orchestrator script you will run.
- **`/data/`**
  - **`/raw/`**: All raw source files (ABR dumps, ATO CTT, Register CSVs) are placed here.
  - **`/processed/`**: All final outputs (Golden Assets, Reports, and Exception Logs) are saved here.
- **`/scripts/`**
  - Contains the 5 master Python scripts that are called by `rebuild.py`.

---

### 2. The Master Python Notebook (The Engine)
*The complete, end-to-end code for the entire project is contained in a single notebook for reference. The operational scripts are broken out for modularity.*

*(This section will contain the full, combined, and commented code from our final 7-cell notebook plan, serving as the ultimate technical reference.)*

---

### 3. The 5 Scripts You Will Use
*The master `rebuild.py` script calls these four scripts in order. They are located in the `/scripts` folder.*

| Script Name | Purpose | Input (from `/data/raw`) | Output (to `/data/processed`) |
| :--- | :--- | :--- | :--- |
| `1_update_entities.py` | **Rebuild the Universe of Identity.** | `abn_bulk_data_all.parquet` | `entity_profiles_v3.parquet` |
| `2_update_obligations.py` | **Rebuild the Universe of Obligations.** | All `YYYY-YY-...-tax-information.xlsx` | `corporate_obligation_log.parquet` |
| `3_update_actions.py` | **Rebuild the Universe of Actions.** | `all-statement-information...csv` | `action_log_final_v2.parquet`, `action_log_exceptions_v2.csv` |
| `4_generate_reports.py` | **Generate Final Primary & Investigative Reports.** | All three golden assets. | `primary_analytical_report.csv`, `multi_pass_investigative_report.csv` |

---

### 4. How to Add New Data (Quarterly/Annually)
*The process is designed to be simple and repeatable.*

1.  **Get the new raw source file(s)** (e.g., the new ATO CTT file for 2024-25).
2.  **Place the new file(s)** into the `/data/raw` directory.
3.  **Run the master script:** `python rebuild.py`.
4.  The script will automatically detect the new file, rebuild the necessary golden assets, and generate the new, updated analytical reports in `/data/processed`.

---

### 5. Emergency Button (If It Breaks)
*If a rebuild fails or the data becomes corrupted, reset to a known good state.*

```bash
# All code is in a version-controlled GitHub repository.
# This command will delete the local copy and clone a fresh, working version.

cd ..
rm -rf ModernSlaveryProject5
git clone https://github.com/your-org/msa-engine.git ModernSlaveryProject5
cd ModernSlaveryProject5
python rebuild.py
