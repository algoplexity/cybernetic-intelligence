

---

### **Definitive Project Rulebook for `ModernSlaveryProject4`**

This document serves as the single source of truth for all business logic, quality gates, and data transformation rules used to construct the foundational universes for this project.

#### **I. Universe of Identity (`entity_profiles_final.parquet` & `active_business_names_final.parquet`)**

This universe answers the question: **"Who are you?"**

**Source Files:**
*   `abn_bulk_data.jsonl` (Primary)
*   `BUSINESS_NAMES_202510.csv` (Enrichment)

**Definitive Rules & Quality Gates:**

1.  **Record Accountability:**
    *   **Rule:** Every single record from `abn_bulk_data.jsonl` must be accounted for.
    *   **Implementation:** Any record that fails processing is quarantined in `identity_exceptions_final.csv` with a specific reason. The final count of `clean + exceptions` must equal the total source record count.

2.  **Identifier Integrity:**
    *   **Rule:** A record is considered valid only if it contains a non-empty, parsable ABN value.
    *   **Implementation:** Records with a missing ABN value are quarantined with `Reason: Missing_ABN_Value`.

3.  **Canonical Formatting:**
    *   **Rule:** All key identifiers (`ABN`, `ACN`) must be stored as uppercase, zero-padded 11-digit strings. All key text fields (`LegalName`, `EntityType_Desc`, etc.) must be stored as uppercase, trimmed strings. All dates must be stored in a consistent `YYYY-MM-DD` format.
    *   **Implementation:** `to_canonical_identifier`, `to_canonical_string`, and `to_canonical_date` functions are applied at the point of extraction.

4.  **Schema Fidelity (ACN):**
    *   **Rule:** A number from the `<ASICNumber>` block is only considered an `ACN` if its `ASICNumberType` attribute is explicitly `'ACN'`.
    *   **Implementation:** The script checks `asic_data.get('@ASICNumberType') == 'ACN'` before extracting the value.

5.  **Trading Name Validity:**
    *   **Rule:** Only business names with a `BN_STATUS` of `'Registered'` are considered active and relevant.
    *   **Implementation:** The `BUSINESS_NAMES_202510.csv` is filtered for `Status == 'Registered'` before the `active_business_names_final.parquet` asset is created.

---

#### **II. Universe of Obligation (`obligation_log_final.csv`)**

This universe answers the question: **"Are you expected to act?"**

**Source Files:**
*   `CorporateTaxTransparency/` folder (ATO CTT reports)
*   `acnc-registered-charities.csv`

**Definitive Rules & Quality Gates:**

1.  **Identity-First Principle:**
    *   **Rule:** An entity can only be included in the Universe of Obligation if it first exists in our validated Universe of Identity.
    *   **Implementation:** Every ABN extracted from both ATO and ACNC sources is checked against the set of valid ABNs from `entity_profiles_final.parquet`. Any record with an unknown ABN is quarantined in `obligation_exceptions_final.csv`.

2.  **Corporate Obligation Definition (ATO):**
    *   **Rule:** A corporate entity is considered to have a presumed obligation for a given year if its `Total_Income` in the corresponding ATO CTT report is **`>= $100,000,000`**.
    *   **Implementation:** The script applies this numerical threshold after loading and cleaning each ATO file.

3.  **Non-Profit Classification (ACNC):**
    *   **Rule:** All ACNC-registered charities are included and classified by their `Charity_Size` ('Large', 'Medium', 'Small'). 'Large' serves as a strong proxy for a reporting obligation.
    *   **Implementation:** The script extracts and retains the `Classification` for all validated charities from the ACNC file.

4.  **Source Data Integrity:**
    *   **Rule:** Records from source files must be structurally sound and contain the necessary columns.
    *   **Implementation:** The script uses a ground-truth driven approach, hardcoded to expect the `'Income tax details'` sheet and specific column names (`'Name'`, `'ABN'`, `'Total income $'`) based on our definitive inspection. Any deviation results in a processing failure, adhering to our "fail-fast" principle.

---

#### **III. Universe of Action (`action_log_final.csv`)**

This universe answers the question: **"What did you do?"**

**Source Files:**
*   `all-statement-information_2025-10-09.csv` (Primary Foundation)
*   `All time data from Register.xlsx` (Strategic Enrichment)

**Definitive Rules & Quality Gates:**

1.  **Identity-First Principle:**
    *   **Rule:** An action can only be logged if it is associated with a valid ABN that exists in our Universe of Identity.
    *   **Implementation:** **(Gate 1)** Any record with a missing, un-extractable, or unknown ABN is quarantined.

2.  **Data Completeness:**
    *   **Rule:** A record must contain a valid `StatementID`, `ReportingYear`, and `Status` to be considered a candidate for the clean log.
    *   **Implementation:** **(Gate 2)** Records with null values in these critical fields are quarantined.

3.  **Logical Uniqueness:**
    *   **Rule:** There can only be one definitive, logical record for any given `(ABN, ReportingYear)`.
    *   **Implementation:** **(Gate 3)** Records belonging to groups with duplicate or contradictory entries for the same entity and year are quarantined.

4.  **Internal Logical Coherence (The "Sanity Check"):**
    *   **Rule:** The data *within* a record must not violate the fundamental business logic of the reporting process.
    *   **Implementation:** **(Gate 4)** Any record that meets the following "impossible" conditions is quarantined:
        *   `Status` is `Published` AND `IsPublishable` is `False`.
        *   `Status` is `Draft` or `Redraft` AND `IsCompliant` is `True`.
        *   `IsPublishable` is `False` AND `IsCompliant` is `True`.

This document now serves as the definitive, auditable rulebook for our entire project.
