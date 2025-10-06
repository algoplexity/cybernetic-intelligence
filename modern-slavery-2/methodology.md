

---

### **White Paper: A Definitive Methodology for Integrated Compliance Analysis**

**Purpose**

This document outlines the definitive, end-to-end methodology for constructing and analysing the Modern Slavery Act compliance landscape. It is the culmination of an iterative process of data discovery, validation, and refinement. This methodology serves as the single source of truth for our analytical process.

**1. Guiding principles**

Our methodology is governed by three core principles:
1.  **Inspect First, Act Second:** We will not make assumptions about any data source. Every file is first subjected to a systematic diagnostic analysis to produce a verified "blueprint" before any data is extracted.
2.  **Entity-Centric, Not Row-Centric:** The fundamental unit of our analysis is the **entity** (identified by its unique ABN). Our data pipeline is designed to correctly handle one-to-many relationships and aggregate data to a clean, one-row-per-entity standard.
3.  **Build Foundational Universes:** We will first transform all raw data sources into three distinct, clean, and purposeful foundational assets. This isolates complexity and creates a clear audit trail.

**2. The methodology: a four-phase process**

Our definitive process consists of four sequential phases, which systematically integrate all eight of our authoritative data sources.

**Phase 1: Build the three foundational universes**
This is the core data engineering phase. Its goal is to transform all raw data into three clean, verified, and de-duplicated foundational assets.

*   **1A: Build the Universe of Identity.**
    *   **Source Files:** `abn_bulk_data.jsonl`, `BUSINESS_NAMES_202510.csv`.
    *   **Logic:** We will process all records from the **ABR Bulk Extract** and the **ASIC Business Names Register**. We will extract every registered business name (legal, trading, etc.) and its associated ABN. This combined data is then cleaned and de-duplicated.
    *   **Output:** `abn_name_lookup.csv` (one row per unique Name-ABN pair).

*   **1B: Build the Universe of Obligation.**
    *   **Source Files:** The six annual `YYYY-YY-corporate-report-of-entity-tax-information.xlsx` files, `acnc-registered-charities.csv`, `COMPANY_202509.csv`.
    *   **Logic:**
        1.  Consolidate all six **ATO Corporate Tax Transparency** reports into a single list of high-revenue corporate entities.
        2.  Join this list with the **ASIC Company Register** to definitively verify the `Type` ('APTY' or 'APUB') for each company.
        3.  Apply the correct, year-specific income threshold ($100M or $200M) based on the verified company type to create a high-confidence list of obligated corporate entities.
        4.  Separately, filter the **ACNC Charity Register** to identify all 'Large' charities, which serve as our proxy for obligated non-corporate entities.
        5.  Combine these two lists to form the complete universe.
    *   **Output:** `obligated_entities.csv` (one unique row per obligated ABN).

*   **1C: Build the Universe of Action.**
    *   **Source File:** `All time data from Register.xlsx`.
    *   **Logic:**
        1.  Start with the raw Modern Slavery Register data.
        2.  Create a **linking table** that correctly resolves the one-to-many relationship between statements and ABNs, using our **Universe of Identity** for name-based ABN repair.
        3.  Join this linking table back to the statement data to associate each ABN with the `Status` and `Period end date` of all its submissions.
        4.  Aggregate this data to determine the *highest* compliance status achieved for each unique ABN for each reporting year.
    *   **Output:** `annual_reporting_log.csv` (one unique row per ABN).

**Phase 2: Build the Master Behavioural File**
The goal of this phase is to integrate our three clean universes into a single, authoritative master file.

*   **Logic:**
    1.  Create a superset list of every unique ABN from the Universe of Obligation and the Universe of Action.
    2.  Use a series of left joins to enrich this master list with all data from the three universes.
    3.  Apply our four-part behavioural classification logic (`Compliant`, `Attempted`, `Initiated`, `Ignored`) to create a `Status_YYYY-YY` column for each year.
*   **Output:** `master_behavioural_file.parquet` (one unique row per ABN).

**Phase 3: Enrichment and profiling**
The goal of this phase is to add the final layers of financial, corporate, and governance risk intelligence to our cohort of non-lodgers.

*   **Source Files:** `master_behavioural_file.parquet`, ATO Tax Transparency reports, `COMPANY_202509.csv`, `clean_associates.csv`, `bd_per_202509.csv`.
*   **Logic:**
    1.  Filter the master file to create our list of non-lodgers.
    2.  Join this list with the consolidated **ATO Tax Transparency** data to append the most recent `TotalIncome`.
    3.  Join with the **ASIC Company Register** to append the current `ASIC_Company_Status`.
    4.  Cross-reference the non-lodgers' directors (from `clean_associates.csv`) with the **ASIC Banned and Disqualified Persons Register** to create the `Has_Banned_Director` flag.
*   **Output:** `enriched_non_lodger_profile.csv` (one unique row per non-lodging ABN).

**Phase 4: Reporting and visualisation**
This is the final, lightweight phase.

*   **Logic:** Use the final, enriched datasets (`master_behavioural_file.parquet`, `enriched_non_lodger_profile.csv`, and the `Taxation Statistics` for context) as the sole inputs. Generate all required summary tables and visualisations for the final executive report.
*   **Output:** The final executive report and all supporting charts.

**3. Conclusion**

This definitive methodology guarantees the integrity, accuracy, and traceability of our findings. By separating the build process into the creation of three distinct, de-duplicated universes before final integration, and by explicitly detailing the role of every single data source, we have a clear and defensible blueprint for our project.

---

### **Appendix A: Foundational Datasets and Evidence Trail (Corrected)**

This appendix details the foundational data assets that were built and integrated to produce the final analytical outcomes of this report. Our methodology was designed to ensure that every finding is traceable back to an authoritative data source and a specific source file.

#### **Summary of Authoritative Data Sources and Files**

The table below lists every external, authoritative data source used in this project, the specific filename(s) processed, and the role each played in building our foundational universes.

| Data Source | Provider | Source Filename(s) | Role in Project |
| :--- | :--- | :--- | :--- |
| **ABN Bulk Extract** | ABR | `abn_bulk_data.jsonl` | **The Foundational "Rosetta Stone".** Forms the core of the **Universe of Identity**. Its comprehensive list of ABNs, ACNs, and all associated business names is the critical link that connects all other datasets and enables our entire analysis. |
| **Corporate Tax Transparency Reports** | ATO | `YYYY-YY-corporate-report-of-entity-tax-information.xlsx` (6 files for 2018-19 to 2022-23) | **Primary Input:** Forms the core of the **Universe of Obligation** by providing a definitive list of high-revenue corporate entities for each year. |
| **Taxation Statistics** | ATO | `tsYY<entity>XX_public.xlsx` (Multiple files for each year from 2018-19 to 2022-23) | **Contextual Analysis:** Used to **quantify the size of the "private company blind spot"**, providing crucial context for the final report. |
| **ACNC Charity Register** | ACNC | `acnc-registered-charities.csv` | **Primary Input:** **Expands the Universe of Obligation** by providing the list of 'Large' charities with a potential reporting requirement. |
| **ASIC Company Register** | ASIC | `COMPANY_202509.csv` | **Verification & Enrichment:** **Verifies the Universe of Obligation** by confirming the 'private' vs. 'public' status of companies. Also enriches the final analysis with company status (e.g., 'Deregistered'). |
| **ASIC Business Names Register** | ASIC | `BUSINESS_NAMES_202510.csv` | **Primary Input:** **Enhances the Universe of Identity** by adding millions of trading names, significantly improving our entity matching capability. |
| **ASIC Banned and Disqualified Persons Register** | ASIC | `bd_per_202509.csv` | **Enrichment:** Enriches the final analysis by providing a **governance risk flag**, linking non-compliant companies to disqualified directors. |
| **Modern Slavery Statements Register** | Internal | `All time data from Register.xlsx` | **Primary Input:** Forms the basis of the **Universe of Action**, providing the raw data on all reporting behaviours (Published, Draft, etc.). |

#### **The Three Foundational Universes**

These authoritative sources were used to construct our three clean, foundational data assets.

*   **1. The Universe of Identity:** A master "phonebook" linking over 15 million business names (from the **ABN Bulk Extract** and ASIC Business Names Register) to their verified ABN. This asset was critical for accurately identifying entities across all other datasets.
*   **2. The Universe of Obligation:** A definitive, year-by-year list of 9,829 unique entities with a confirmed legal obligation to report, built from the ATO Corporate Tax Transparency and ACNC Charity Register data, and verified using the ASIC Company Register.
*   **3. The Universe of Action:** A complete, year-by-year log of every action (Published, Draft, Redraft) taken by every entity in the Modern Slavery Register. This log, our `annual_reporting_log.csv`, was built by cleaning the raw Register data and repairing missing ABNs using our Universe of Identity.

By systematically integrating these three foundational universes, we were able to build the **Master Behavioural File** that underpins all the findings in this report, ensuring that every conclusion is fully traceable and evidence-based.
