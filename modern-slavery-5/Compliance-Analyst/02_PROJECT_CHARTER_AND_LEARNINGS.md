# Project Charter & Learnings: Modern Slavery Act Compliance Engine

**Purpose:** This document contains the full narrative, justification, and key learnings from the Ground-Zero Rebuild of the Modern Slavery Act Compliance Engine. It explains the "why" behind every critical decision, providing the deep context needed for future development, auditing, and strategic planning.

**Audience:** Project Managers, Future Developers, Auditors, and key Business Stakeholders.

---

### 1. The Ultimate Goal and Final Strategy

**The Problem:** Initial attempts to build this compliance engine were plagued by repeated, difficult-to-diagnose failures. Final reports were found to be inaccurate, with compliant entities being incorrectly flagged as non-compliant. A "Ground-Zero Rebuild" was initiated with the core principle of creating a fully defensible, auditable, and repeatable process.

**The Final, Successful Strategy:** The project concluded that a single report could not be both perfectly auditable (using only high-certainty data) and perfectly complete (accounting for flawed source data). The definitive solution is a **Two-Report System**:

1.  **The Primary Report (The Auditor):** A strict, high-certainty report built by starting with the universe of financially obligated entities and joining it to a clean, validated universe of compliance actions using a verifiable `ABN`-to-`ABN` link. This report is the source of truth for auditable compliance, but it is known to produce "false negatives" when source data is flawed.

2.  **The Investigative Report (The Detective):** A sophisticated, multi-pass report that re-examines the obligated cohort. It uses a hierarchical `ABN -> Fuzzy Name -> ACN` matching logic against the **raw source data** to find and prove the existence of these "false negatives" (like the "Allianz" case), providing a more complete, real-world picture of compliance.

The true compliance landscape is only understood by using these two reports in conjunction.

---

### 2. The Unbreakable Rules (Our Hard-Won Lessons)

*These are the critical principles that govern the entire engine. They were discovered through rigorous testing and failure analysis. Adherence to these rules is non-negotiable for maintaining the defensibility of the outputs.*

#### **Lesson 1: The ABN is the Only True Key. The ACN is Trash.**
- **The Finding:** The Australian Business Number (ABN) is the only reliable, canonical, and universal identifier for matching entities across all datasets. The Australian Company Number (ACN), represented as `ASICNumber` in the source data, is structurally optional, frequently invalid, and must not be used for high-confidence joins.
- **The Evidence:**
    1.  **The ABR XSD Schema:** Our deep dive into the official schema proved that the `<ASICNumber>` element is an optional, top-level attribute of a primary entity record (`<ABR>`). It is structurally forbidden from appearing within associated `<OtherEntity>` records (like trading names). This means, by design, most records do not have an ACN.
    2.  **The Exception Log:** The `abn_acn_exceptions_log.csv` contains **2.6 million records** where the provided `ASICNumber` failed a basic check-digit validation algorithm, proving the source data quality is extremely poor.
    3.  **The Validation Scripts:** Our own validation scripts proved that the ABNs in the source data were **100% mathematically valid**.
- **The Principle:** All high-certainty joins **must** use the ABN. The ACN should only be used as a final, low-confidence clue in a multi-pass investigative search and its results must be flagged as such.

#### **Lesson 2: The Source Statement Data is Critically Flawed.**
- **The Finding:** The raw source file for compliance actions (`all-statement-information...csv`) is riddled with severe, compounding data quality issues that make simple processing impossible.
- **The Evidence:** Our final `action_log_exceptions_v2.csv` provides a complete audit trail of these flaws, which would have otherwise corrupted our analysis. The key issues are:
    1.  **Missing Primary ABNs (~10%):** The primary, structured `ABN` field is often null, even when ABNs are present as plain text within the `ReportingEntities` string. This was the root cause of the Allianz "false negative."
    2.  **Non-Annual Periods:** A significant number of statements do not conform to the Act's "annual accounting period" rule, with durations ranging from 0 to over 1000 days.
    3.  **Duplicate Statement IDs:** The same unique statement ID is sometimes assigned to multiple, different entities, making the records ambiguous.
- **The Principle:** A rigorous, multi-stage Quality Control (QC) process is **non-negotiable** for building the Universe of Actions. Every record must be validated for ABN presence, date integrity, annual period duration, and uniqueness before it can be considered "defensible." All failing records must be quarantined to a detailed exception log.

#### **Lesson 3: An Action Can Span Multiple Years. You *Must* Explode It.**
- **The Finding:** A single compliance statement with a non-standard reporting period (e.g., a calendar year) can satisfy an entity's legal obligation for **multiple** Australian financial years.
- **The Evidence:** The Allianz statement for `1 Jan 2022 - 31 Dec 2022` correctly maps to both the `2021-22` and `2022-23` financial years. Our initial, flawed action log failed to create the `2022-23` record, causing our Primary Report to incorrectly label Allianz a "Non-Lodger."
- **The Principle:** All statement data **must** be "exploded" based on its `PeriodStart` and `PeriodEnd` dates. This involves a calculation to create a separate, canonical `ReportingYear` record for every single financial year a statement touches. This is the single most important transformation for ensuring analytical accuracy. Our "Control Total Framework" was developed to mathematically prove this explosion is performed without data loss.

#### **Lesson 4: The Identity Universe is Broader Than Just "Legal" Names.**
- **The Finding:** A massive number of key entities, particularly those from the ATO's high-revenue list, do not have a formal `'LGL'` (Legal) name record in the ABR data. They exist only under a `'MN'` (Main Name).
- **The Evidence:** Our initial "fail-fast" merge prototype failed completely, with the `Identity prototype shape: (0, 7)`. This proved that our first, schema-pure Identity Universe was missing the very entities we needed to analyze.
- **The Principle:** To be comprehensive, our Identity Universe **must** be built using a prioritized fallback system: use the `'LGL'` name if it exists, otherwise use the `'MN'` name. To maintain defensibility, the final `entity_profiles_v3.parquet` file **must** contain the `Name_Source` column (`'Legal'` or `'Main'`) to provide a transparent audit trail for the origin of every name.

---

### 3. A Guide to the Final Outputs (The Audit Trail)
*Every file in the `/data/processed` folder has a specific, defensible purpose.*

- **`primary_analytical_report.csv`:** The Auditor's Report. This is the high-certainty output of the project. Its "Non-Lodger" list is the definitive starting point for all compliance investigations. The ~7,000 records with a missing `LegalName` represent the real, quantified data gap between the ATO's taxation list and the ABR's primary registration data—a critical finding in itself.

- **`multi_pass_investigative_report.csv`:** The Detective's Report. This is the necessary companion to the primary report. Its purpose is to take the "Non-Lodger" list and find false negatives. The multi-column output (`ABN_Match`, `Name_Match`) provides a clear, defensible audit trail for the confidence of each match found.

- **`action_log_exceptions_v2.csv`:** The proof of our rigor. This file contains every statement record that was defensibly excluded from our clean `action_log`. The `QuarantineReason` column explains exactly why each record was deemed untrustworthy. The reason the Allianz statement for 2022-23 is not in the primary report's action log is *because* its flawed source record is in this file.

- **`secondary_unmatched_exceptions.csv`:** The end of the road for automated analysis. This file lists all the entities from the `action_log_exceptions_v2.csv` that we attempted to rescue via name-matching, but for which no defensible match could be found. These records require manual human investigation.
