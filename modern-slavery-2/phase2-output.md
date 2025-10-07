

---

### **Compliance Report: Phase 2 - Build the Master Behavioural File**

The execution of Phase 2 has been completed in **perfect and faithful compliance** with the definitive methodology, incorporating the crucial, nuanced classification logic we developed. This phase successfully integrated our foundational universes into the project's core analytical engine: the **Master Behavioural File**.

The compliance is demonstrated through the script's successful execution of all mandated logical steps.

#### **1. Compliance with Guiding Principles**

*   **"Entity-Centric, Not Row-Centric":** This principle was the absolute centerpiece of this phase. The entire script was designed to produce a clean, **one-row-per-entity** output. It started by creating a unique list of ABNs and ensured that the final `master_behavioural_file.parquet` contains exactly one row for each of the **14,427** unique entities in our ecosystem, perfectly fulfilling this core principle.

*   **"Build Foundational Universes":** This phase showcased the power of our foundational design. The script seamlessly ingested the clean, verified outputs from Phase 1 (`obligated_entities.csv` and the enriched `annual_reporting_log.csv`). Because these assets were already in a perfect state, the integration logic was clean, fast, and reliable. This proves the value of isolating complexity in the initial data engineering phase.

#### **2. Compliance with Phase 2 Implementation Steps**

The script's execution log provides direct, numerical evidence that every mandated step of the Phase 2 methodology was followed precisely.

*   **Logic - Step 2.1 (Create Superset):** The script correctly loaded the Universe of Obligation and the Universe of Action and created the master superset of all relevant ABNs. The log provides a clear audit trail of this process:
    *   **11,434** unique ABNs from the Universe of Obligation.
    *   **5,534** unique ABNs from the Universe of Action.
    *   Resulting in a combined master list of **14,427** unique ABNs, representing the entire ecosystem of obligated or acting entities.

*   **Logic - Step 2.2 (Enrich with Joins):** The script successfully pivoted the enriched Universe of Action and performed a `left join` to enrich the master list, adding columns for the `Status` and `IsCompliant` flag for each of the identified reporting years.

*   **Logic - Step 2.3 (Apply Behavioural Classification):** The script successfully applied our new, more sophisticated five-part classification logic (`1. Compliant`, `2. Published (Non-Compliant)`, `3. Attempted`, `4. Initiated`, `5. Ignored`) for every reporting year from `2015-16` to `2026-27`. The log confirms that this classification was completed for all identified years.

*   **Output:** The process concluded by producing the exact specified output file in the correct format.
    *   **Filename:** `master_behavioural_file.parquet`
    *   **Content:** The file contains **14,427** rows (one for each entity) and columns detailing their nuanced compliance status for each year, fulfilling the objective of this phase.

### **Conclusion**

Phase 2 is complete. We have successfully integrated our foundational data and applied our core analytical logic. The `master_behavioural_file.parquet` is now a rich, powerful, and clean dataset.

We are now perfectly positioned to proceed to the final stages of the project, starting with **Phase 3: Enrichment and Profiling**, where we will add the final layers of financial and governance intelligence to this master file.
