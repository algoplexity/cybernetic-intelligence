This is the complete picture. With this final set of screenshots, my understanding of the BRAIN Labs platform and its intended workflow is comprehensive. We have moved from basic discovery to expert-level data handling and are now fully equipped to formulate an intelligent, effective alpha expression.

The new examples provide the final, most critical lessons: how to handle real-world data imperfections like inconsistent units, sparse data, and extreme outliers.

### The Complete, Intelligent AI Workflow for Alpha Creation

Here is the final, multi-phase strategy that synthesizes every lesson we have learned. This is the exact process our AI must follow.

---

### Phase 1: Scoping & Discovery (The "What")

This is the foundational step we established first.

1.  **Find Categories:** Use `brain.get_data_categories()` to see the high-level data types (e.g., `pv`, `fundamental`, `sentiment`).
2.  **Find Datasets:** Use `brain.get_data_sets(category=...)` to find specific data products (e.g., `pv1`, `mdl26`).
3.  **Find Fields:** Use `brain.get_data_fields(data_set=...)` to get a list of all usable field IDs (e.g., `close`, `revenue`, `mdl26_arm_score`).
4.  **Check Field Quality:** For a promising field, inspect its `DataFieldDetails` (via the `.data` attribute) to check its `coverage`, `alpha_count`, and `user_count` for the target universe.

**Outcome:** A list of high-potential `field_ids` to investigate further.

---

### Phase 2: Deep Field Analysis & Profiling (The "Why")

This is the most critical phase, where we apply the lessons from the platform's advanced examples to avoid common pitfalls. For each candidate field:

1.  **Download the Data:** Get the raw time-series data into a DataFrame.
    *   **Code:** `df = brain.get_data_frame(brain.get_data_field('field_id'))`

2.  **Analyze Data Availability:**
    *   **Question:** Is the data sparse? Does it arrive periodically?
    *   **Method:** Use `visualizations.plot_coverage(df)`. Look for sinusoidal patterns as seen in the `etz_revenue` example.
    *   **Insight:** If so, `ts_backfill` will be necessary to create a continuous signal.

3.  **Analyze Data Meaning:**
    *   **Question:** What do zero values mean? Are they "no data" or a meaningful "zero"?
    *   **Method:** Examine the `percent_zero` vs. `percent_nan` lines on the coverage plot.
    *   **Insight:** If `percent_zero` is high (like the `dividend` example), it means `0` is a real value and may need to be converted to `NaN` with `to_nan()` before backfilling to prevent false turnover.

4.  **Analyze Data Scale and Units:**
    *   **Question:** What are the units? Is this a raw value, a rank, or a score?
    *   **Method:** Use `visualizations.plot_distribution()` and `visualizations.plot_instrument_values()`.
    *   **Insight:** This will reveal if fields are in thousands vs. billions (`revenue` vs. `fnd3_Q_revenue`) or if they are on completely different scales (raw `revenue` vs. ranked `mdl26_revenue`). **Fields with incompatible scales cannot be combined without normalization.**

5.  **Analyze Outliers:**
    *   **Question:** Are there extreme, transient outliers?
    *   **Method:** Use `visualizations.plot_descriptive_stats()`. Look for sudden, sharp spikes in the `min` or `max` lines.
    *   **Insight:** As shown with `etz_revenue`, these outliers are not persistent and must be handled by date, not by instrument. The `winsorize` operator is ineffective here.

---

### Phase 3: Intelligent Preprocessing (The "How")

Based on the analysis from Phase 2, we build a specific preprocessing pipeline for our alpha expression. The order of operations is critical.

1.  **Handle Extreme Outliers:** If transient outliers were detected, nullify them first.
    *   **Pattern:** `(field > ts_max(field, 252) ? nan : field)`

2.  **Handle Meaningful Zeros:** If necessary, convert meaningful zeros to `NaN`.
    *   **Pattern:** `to_nan(field)`

3.  **Normalize Units:** If combining fields with different raw units, apply scaling factors.
    *   **Pattern:** `fnd3_Q_revenue / 10^6`

4.  **Backfill Data:** If data is sparse or periodic, fill in the missing values.
    *   **Pattern:** `ts_backfill(field, 60)`

5.  **Normalize for Alpha Combination:** Before combining preprocessed fields into a final signal, ensure they are on a common scale.
    *   **Pattern:** `ts_zscore(field, 252)` or `group_rank(field, subindustry)`

---

### Phase 4: Final Formulation & Simulation

Now, using the fully preprocessed fields, we can confidently construct our alpha and test it.

**Example: A Robust `etz_revenue` Alpha**

```
# Step 1: Handle outliers
etz_no_outliers = (etz_revenue > ts_max(etz_revenue, 252) ? nan : etz_revenue);

# Step 2: Backfill the now-cleaner data
etz_backfilled = ts_backfill(etz_no_outliers, 65);

# Step 3: Normalize the signal by a fundamental factor (e.g., assets) and then by group
final_alpha = group_extra(ts_zscore(etz_backfilled / assets, 252), 1, subindustry);
```

This expression, informed by every lesson from the platform, is infinitely more robust and likely to perform better than a naive `etz_revenue / assets` formula.

We have successfully completed the journey from novice discovery to expert-level alpha formulation. We are ready to create.
