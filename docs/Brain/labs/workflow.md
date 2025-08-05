

The `how_to_use_ACE.txt` notebook and its associated documentation reveal that the platform provides **two distinct but complementary sets of APIs**:

1.  **The `brain` Client:** A low-level, interactive toolkit for discovery and deep analysis.
2.  **The `ace_lib` & `helpful_functions`:** A high-level, automated toolkit for systematic generation, bulk simulation, and submission.

Think of it as having two toolboxes:

---

### Toolbox 1: The `brain` Client (The "Explorer & Scientist's Kit")

*   **Purpose:** To perform the deep, investigative work on a *single idea*. This is what we have been using so far.
*   **Key Functions:**
    *   `brain.get_data_categories()`, `brain.get_data_sets()`, `brain.get_data_fields()` for metadata discovery.
    *   `brain.get_data_frame()` to download raw data for profiling.
    *   `visualizations.plot_...()` to understand the quirks of a specific field.
    *   `brain.simulate_alpha()` to run a quick test on a single expression.
*   **When to Use:** During the initial "Analysis & Profiling" phase of our plan. It's for answering questions like, "What does the `volume` data *really* look like?" or "Does `returns` have outliers?"

### Toolbox 2: The ACE Library (The "Alpha Factory & Manager")

*   **Purpose:** To take a promising alpha *template* and efficiently generate, test, improve, and submit many variations.
*   **Key Functions:**
    *   `hf.get_datasets()` and `hf.get_datafields()` to programmatically select data based on quality metrics (e.g., `coverage > 0.8`).
    *   `ace.generate_alpha()` to package an expression with its simulation settings.
    *   `ace.simulate_alpha_list_multi()` to run **dozens of simulations concurrently**.
    *   `hf.prettify_result()` and `hf.concat_is_tests()` to analyze and compare the results of all simulations at once.
    *   `ace.submit_alpha()` to formally submit a finished, validated alpha.
*   **When to Use:** During the "Formulation, Simulation, & Iteration" phases of our plan, especially when we want to test multiple variations of an idea.

---

### The New, Integrated Workflow

This new information doesn't invalidate our previous plan; it supercharges it. The `ace_lib` automates the final, large-scale steps.

Here is the updated, definitive plan that integrates both toolkits:

**Phase 1: Deep Dive with the `brain` Client (Unchanged)**
*   Use the `brain` client and `visualizations` to deeply analyze our candidate fields (`returns`, `volume`). This gives us the core insights for our "intelligent" preprocessing logic (e.g., handle outliers, normalize volume).

**Phase 2: Systematic Generation with `ace_lib` (The New Step)**
*   Instead of writing just one "intelligent" alpha, we will generate a list of expressions to test several variations of the idea.
    *   **Example:** We can test different lookback windows.
    ```python
    # Our core idea for a clean signal
    preprocessing_logic = """
    r_clean = (abs(returns) > 0.5 ? nan : returns);
    vol_surge = (volume / ts_mean(volume, 20));
    vol_surge_clean = (vol_surge > 10 ? 10 : (vol_surge == inf ? nan : vol_surge));
    """
    
    # Generate variations with different momentum periods
    expression_list = []
    for period in [5, 10, 20, 40]:
        expression = f"""
        {preprocessing_logic}
        raw_alpha = ts_rank(r_clean, {period}) * ts_rank(vol_surge_clean, {period});
        group_rank(raw_alpha, subindustry)
        """
        expression_list.append(expression)
    ```

**Phase 3: Bulk Simulation & Analysis with `ace_lib`**
*   **Generate Alpha Objects:** Use `ace.generate_alpha()` on our `expression_list`.
*   **Simulate in Parallel:** Use `ace.simulate_alpha_list_multi()` to run all simulations at once.
*   **Analyze Results:** Use `hf.prettify_result()` to get a summary DataFrame. We can then easily sort by `fitness` or `sharpe` to find the best-performing variation.

**Phase 4: Improvement & Submission with `ace_lib`**
*   **Select the Winner:** Pick the alpha with the best metrics from the summary table.
*   **Check Submission Readiness:** Use `hf.concat_is_tests()` and other checks (`check_self_corr_test`) to ensure the winning alpha passes all the platform's quality gates.
*   **Submit:** Use `ace.submit_alpha()` with the winning `alpha_id`.

This is the true power of the platform. We use the `brain` client for surgical analysis and insight, and then we use the `ace_lib` as a powerful factory to scale up our testing and find the optimal version of that insight.

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
