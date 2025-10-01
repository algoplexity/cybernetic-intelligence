
The goal is to consolidate the information from these four disparate sources into a single, reliable **Master Entity File**. This file will serve as our source of truth, containing one unique row per entity (identified by its ABN) and summarizing its reporting status and key attributes.

---

### **Work Plan: Creating the Foundational Master Entity File**

This plan is structured in three phases:

1.  **Phase 1: Individual File Ingestion and Cleaning:** Each file will be processed independently to handle specific data quality issues.
2.  **Phase 2: Data Consolidation and Enrichment:** The cleaned data will be merged to build a comprehensive profile for each entity.
3.  **Phase 3: Finalization and Validation:** The consolidated dataset will be finalized with new, insightful features and validated for accuracy.

---

#### **Phase 1: Individual File Ingestion and Cleaning**

The objective here is to standardize key fields and correct errors within each source file before attempting to merge them.

| Task ID | File Name | Tab/Data | Cleaning & Preparation Steps |
| :--- | :--- | :--- | :--- |
| **P1-A** | `All time data from Register.xlsx` | **Statements** | 1.  **Standardize ABNs:** Extract and clean ABNs from the `Reporting entities` column. ABNs will be converted to a uniform 11-digit string format, stripping any spaces or special characters and padding with leading zeros where necessary. <br> 2.  **Parse Dates:** Convert `Submitted`, `Date published`, `Period start date`, and `Period end date` columns to a consistent `YYYY-MM-DD` datetime format. Invalid date entries will be flagged. <br> 3.  **Standardize Revenue:** Convert the `Revenue` column to a numeric integer format (e.g., "500-600M" becomes an average or is categorized). <br> 4.  **Deduplication:** Check for and remove any fully duplicate rows representing the same statement submission. |
| **P1-B** | `All time data from Register.xlsx` | **Entities** | 1.  **Standardize ABNs:** Ensure the `ABN` column conforms to the 11-digit string format. <br> 2.  **Standardize Company Names:** Clean the `Company name` column by converting to a consistent case (e.g., uppercase), trimming leading/trailing whitespace, and standardizing common suffixes (e.g., "PTY LTD" to "PTY. LTD."). |
| **P1-C** | `ato_tax_transparency_non_lodger.xlsx` | **Non-Lodger** | 1.  **Standardize ABNs:** Clean the `ABN` column to the standard 11-digit string format. <br> 2.  **Standardize Company Names:** Apply the same cleaning rules as P1-B to the `Entity Name` column. <br> 3.  **Standardize Financials:** Ensure the `Total Income` column is a clean numeric type. <br> 4.  **Column Pruning:** Select a core set of relevant columns (e.g., ABN, Entity Name, Total Income, State, ASX status, Industry) to create a focused master entity profile. |
| **P1-D** | `lodge_once.csv` & `lodge_once_cont.xlsx` | **lodge\_once** | 1.  **Merge Datasets:** Perform an inner join of the two files on the `abn` column to create a single, unified dataset for single-lodger entities. <br> 2.  **Standardize ABNs:** Clean the `abn` column in both files before merging. <br> 3.  **Parse Dates:** Convert all date-related columns (`last_submission_dttm`, `expected_due_date`, etc.) to the standard `YYYY-MM-DD` format. |

---

#### **Phase 2: Data Consolidation and Enrichment**

The objective is to create the master file by merging the cleaned datasets and adding summary information.

| Task ID | Action | Methodology |
| :--- | :--- | :--- |
| **P2-A** | **Establish Base Master Entity File** | Use the cleaned `Non-Lodger` data (from P1-C) as the foundational list of all potential reporting entities. This file will be our "universe" of entities to track. |
| **P2-B** | **Aggregate Submission History** | From the cleaned `Statements` data (from P1-A), create a summary table aggregated by ABN. This summary will include:<br>- `num_statements_submitted`<br>- `first_submission_date`<br>- `last_submission_date`<br>- A list of all reporting years. |
| **P2-C** | **Merge Submission History** | Perform a **left join** from the `Master Entity File` (P2-A) to the aggregated submission history (P2-B) using the ABN as the key. |
| **P2-D** | **Create Reporting Status Flags** | Based on the merge in P2-C, create new columns in the `Master Entity File`:<br>- `has_ever_reported`: A boolean flag (True/False). If the submission history columns are null, this will be False.<br>- `is_multi_year_reporter`: A boolean flag indicating if the entity has reported for more than one reporting cycle. |
| **P2-E** | **Enrich with Single-Lodger Data** | Perform a **left join** from the `Master Entity File` to the cleaned and merged `lodge_once` data (from P1-D) on ABN. This will add the detailed compliance data (e.g., `nc_index`, `repeat_nc`) for the relevant subset of entities. |

---

#### **Phase 3: Finalization and Validation**

The objective is to produce a clean, documented, and ready-to-use master file for the next stage of the investigation.

| Task ID | Action | Methodology |
| :--- | :--- | :--- |
| **P3-A** | **Calculate "Days Since Last Submission"** | Create a new column, `days_since_last_submission`, by calculating the difference between today's date and the `last_submission_date`. This will be a key indicator for identifying potential non-lodgers in the current cycle. |
| **P3-B** | **Final Schema and Data Dictionary** | Organize the columns in the final `Master Entity File` into logical groups (Entity Identifiers, ATO Profile, Reporting Summary, Compliance Details). Create a comprehensive data dictionary explaining each field, its data type, and its origin. |
| **P3-C** | **Output and Validation** | 1.  Generate the final `Master Entity File` as a clean CSV or Excel file. <br> 2.  Provide a summary report detailing the cleaning process, including the number of records processed from each source, the number of ABNs successfully matched, and a list of any data quality issues that require further manual review. |

Upon completion of this work plan, we will possess a single, high-integrity **Master Entity File**. This foundational asset will be pivotal for efficiently and accurately identifying entities that have failed to lodge their modern slavery statements, forming the basis for all subsequent analysis and compliance actions.
