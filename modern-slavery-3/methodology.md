
---

### **The Definitive Project Methodology: From Raw Data to Insight**

**Project Goal:** To create a reproducible, auditable, and comprehensive analysis of the Modern Slavery Act compliance ecosystem.

**Core Principle:** "Inspect First, Act Second." Every script is built upon a definitive blueprint of its source data, and every foundational asset is validated before it is used.

**Required Raw Inputs (to be placed in `ModernSlaveryProject2/`):**
1.  `abn_bulk_data.jsonl` (ABR)
2.  `BUSINESS_NAMES_202510.csv` (ASIC)
3.  `COMPANY_202509.csv` (ASIC)
4.  `acnc-registered-charities.csv` (ACNC)
5.  `bd_per_202509.csv` (ASIC)
6.  `ato_tax_transparency_non_lodger.xlsx` (Internal)
7.  `lodge_once_cont.xlsx` (Internal)
8.  `All time data from Register.xlsx` (Internal, from the Register)
9.  All `YYYY-YY-corporate-report-of-entity-tax-information.xlsx` files (ATO) in a `CorporateTaxTransparency/` subfolder.

---

### **The End-to-End Process: A Sequence of Five Definitive Scripts**

The entire analysis is now a clean, sequential process of five definitive scripts.

**Script 1: The "Golden" Entity Profile Generator**
*   **Purpose:** To create our definitive "Universe of Identity."
*   **Core Input:** `abn_bulk_data.jsonl`
*   **Process:** Reads the massive JSONL file in robust, memory-safe chunks. It extracts the 10+ key features for every entity (ABN, ACN, LegalName, EntityType, Status, Dates, Location, etc.) and applies our "canonical formatting" toolbox to produce a perfectly clean profile for all 19.5 million entities.
*   **Output:** The first "golden" asset: **`entity_profiles.parquet`**.

**Script 2: The Rich Corporate Obligation Log Generator**
*   **Purpose:** To create our "gold standard" "Universe of Corporate Obligation."
*   **Core Inputs:** The raw ATO Tax Transparency files and our new `entity_profiles.parquet`.
*   **Process:** It loads the `entity_profiles.parquet` to create an `EntityType` lookup. It then processes all ATO files, applying our proven, year-specific `$100M/$200M` threshold logic to identify and record every single instance of a proven corporate obligation.
*   **Output:** The second "golden" asset: **`corporate_obligation_log.csv`**, a rich, self-documenting file.

**Script 3: The Intelligent Action Log Generator**
*   **Purpose:** To create our definitive "Universe of Action."
*   **Core Input:** The `All time data from Register.xlsx`.
*   **Process:** It reads the `'Statements'` sheet and intelligently separates the data. It isolates the ~2,660 "factually impossible" contradictory records into an exception file for human review, ensuring the final asset is 100% clean and logically consistent.
*   **Outputs:**
    *   The third "golden" asset: **`action_log.csv`** (containing `ABN`, `ReportingYear`, `Status`, `IsCompliant`).
    *   An auditable exception file: `action_log_exceptions.csv`.

**Script 4: The Governance Log Generator**
*   **Purpose:** To create our final foundational "Universe of Governance."
*   **Core Inputs:** `ato_tax_transparency_non_lodger.xlsx` and `lodge_once_cont.xlsx`.
*   **Process:** It extracts the `'associates'` sheets from both files, combines them, and applies our canonical formatting to produce a clean list of all known director-company relationships.
*   **Output:** The fourth "golden" asset: **`governance_log.csv`**.

**Script 5: The TRUE Master Analytical File Generator**
*   **Purpose:** To integrate all our foundational work into the single source of truth.
*   **Core Inputs:** Our four "golden" assets: `entity_profiles.parquet`, `corporate_obligation_log.csv`, `action_log.csv`, and the ACNC charity register (for size).
*   **Process:** This is the heart of the project. It creates a comprehensive "long" format DataFrame, one row for every entity-year in our ecosystem. It then merges all the facts from our golden assets and applies our final, perfected 23-part classification logic to generate the definitive `Stakeholder_Status` for every entity in every year.
*   **Output:** The final, definitive "golden" asset: **`master_analytical_file_v2.parquet`**.

This five-script process is the definitive, reproducible, and validated methodology for this project. Any final report or visualization can now be generated with a simple, lightweight script that uses the `master_analytical_file_v2.parquet` as its single, trusted input.
