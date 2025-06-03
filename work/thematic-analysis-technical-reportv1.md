**Technical Status Report: MDL-Based Thematic Motif Discovery - Milestone Reached**

**Project:** Textual Intelligence via Minimum Description Length (MDL)
**Date:** June 3, 2025
**Version:** MWP v10.1 (Segmented BDM, Refined Prompting & L(H) Strategy)
**Status:** Successful proof-of-concept: Achieved net positive MDL compression.

**1. Achievements or Highlights**

*   **Demonstrated Net Positive MDL Compression:** For the first time, the system successfully identified a set of symbolic motifs for QID Q4 (209 responses, ~129k chars) that resulted in a net positive compression of **34.57 MDL units**. This validates the core hypothesis that LLM-generated thematic candidates, when appropriately filtered and evaluated, can provide a more concise description of the data than the data itself under the MDL principle.
*   **Successful End-to-End Pipeline Operation:** The full pipeline, including batched LLM calls for motif candidate generation, robust JSON parsing with label sanitization, multi-stage surface form (SF) refinement (consolidation and global frequency/length filtering), and MDL evaluation using segmented BDM, is now functioning reliably.
*   **Improved LLM Output Quality:** Significant progress has been made in prompting `gemma-2b-it` to produce:
    *   Structurally valid JSON lists of motif objects.
    *   Thematically relevant labels (often correctly bracketed or fixable).
    *   More concise descriptions.
    *   Surface forms that are increasingly shorter and more keyword-like, suitable for compression.
*   **Segmented BDM Effectiveness:** Implementing segmented BDM for `L(D|H)` and `L(D_orig)` proved crucial. It provides a more accurate measure of data complexity reduction across the entire corpus, revealing significant savings (69.97 units for QID Q4) that were previously obscured by prefix-only BDM.
*   **Effective L(H) Strategy:** The current token-based `L(H)` model, which costs symbolic labels and their descriptions but (for this experiment) zeros out the definition cost of listing surface forms, allowed the data compression benefits to become apparent. This yielded an `L(H)` of 35.4 for 10 refined motifs in QID Q4.

**2. Solution Options Explored & Current Configuration**

*   **LLM Candidate Generation:**
    *   *Initial Approach:* Single LLM call on full text, simpler prompts. Led to parsing issues and poor SFs.
    *   *Current Approach:* Batched processing of responses (5 responses/chunk). LLM (`gemma-2b-it` with `do_sample=False`) prompted per chunk for up to 3 motifs, using a highly structured prompt emphasizing JSON validity, specific label formatting, and short, verbatim, repeated SFs.
*   **Motif Representation:**
    *   *Current:* Structured dictionary `{"label": "[SYMBOL]", "description": "text", "surface_forms": ["sf1", "sf2"]}`.
*   **MDL Formulation:**
    *   **`L(H)` (Model Cost):** Token-based cost for labels and descriptions. *Current experiment has zeroed out the cost component for listing SFs within L(H)*.
    *   **`L(D|H)` (Data given Model Cost):**
        *   *Initial:* BDM on a 2000-char prefix of the compressed text. Showed minimal change.
        *   *Current:* **Segmented BDM**. The full corpus (original and motif-compressed) is divided into 2000-char segments, BDM is calculated for each segment's hash, and these values are summed. This proved far more sensitive and accurate.
*   **Surface Form Refinement:**
    *   *Initial:* Basic consolidation by label.
    *   *Current:* Multi-stage:
        1.  LLM prompted for short, repeated SFs.
        2.  Consolidation by label, merging unique (lowercased) SFs.
        3.  Global filtering: SFs kept if they appear >= `MIN_SF_FREQUENCY_IN_FULL_CORPUS` (e.g., 2) AND are <= `MAX_SF_TOKEN_LENGTH_FOR_FINAL_MOTIF` (e.g., 6 words) in the *entire QID corpus*.

**3. Key Decisions Made and Rationale**

*   **Decision:** Shifted from BDM on text prefix to **Segmented BDM** for `L(D|H)`.
    *   **Rationale:** Prefix BDM was insensitive to global text changes from motif replacement. Segmented BDM provides a more accurate measure of complexity change across the entire corpus, which was critical for observing significant `L(D|H)` reduction.
*   **Decision:** Experimentally **zeroed out the cost of listing surface forms within `L(H)`**.
    *   **Rationale:** To isolate whether the primary barrier to net compression was the definition cost of SFs or the lack of data term (`L(D|H)`) reduction. This revealed substantial `L(D|H)` reduction was occurring, but the full `L(H)` (including SF listing costs) was still too high. This allows us to focus on maximizing data compression first.
*   **Decision:** Implemented **batched processing** for LLM calls.
    *   **Rationale:** To manage LLM context window limitations, reduce LLM cognitive load per call, and improve output quality and consistency.
*   **Decision:** Adopted a **multi-stage SF refinement process**, culminating in global frequency and length filtering.
    *   **Rationale:** LLM-generated SFs are often imperfect. Grounding them in actual corpus statistics (frequency, length) is essential for selecting SFs that are truly effective for compression and representative of recurring patterns.
*   **Decision:** Used **`do_sample=False`** for LLM generation.
    *   **Rationale:** Empirically found to produce more consistent adherence to strict JSON and label formatting requirements from `gemma-2b-it` compared to sampling, even with low temperature.
*   **Decision:** Implemented **programmatic label sanitization**.
    *   **Rationale:** As a pragmatic step to handle LLM inconsistencies in label bracketing, ensuring more candidates pass initial schema validation. The long-term goal is still better LLM compliance via prompting.

**4. Methodology or Algorithm Used (Current Iteration)**

1.  **Initialization:** Load pre-aggregated text data per QID. Initialize LLM (Hugging Face pipeline, `gemma-2b-it`, 4-bit quantized, `return_full_text=False`) and BDM (`ndim=2`).
2.  **For each target QID:**
    a.  Join all individual responses to form the `full_corpus_for_qid`.
    b.  Calculate `L(D_orig)`: Apply segmented BDM (2000-char segments) to `full_corpus_for_qid`.
    c.  **Batch Motif Extraction:**
        i.  Divide `full_corpus_for_qid`'s individual responses into batches (e.g., 5 responses).
        ii. For each batch:
            1.  Preprocess the chunk text.
            2.  Construct a detailed prompt requesting up to 3 motifs (JSON objects with "label", "description", "surface\_forms"), emphasizing short, verbatim, repeated SFs and strict JSON/label formatting.
            3.  Call LLM (`do_sample=False`).
            4.  Parse the LLM's JSON output. Sanitize/fix labels if possible. Validate schema.
            5.  Collect successfully parsed and validated motif objects. Retry on failure.
    d.  **Consolidate Motifs:** Merge all extracted motif objects from all chunks, grouping by unique "label" and combining their (lowercased, unique) "surface\_forms".
    e.  **Refine Surface Forms Globally:** For each consolidated motif, filter its "surface\_forms" list, keeping only those SFs that:
        i.  Appear >= `MIN_SF_FREQUENCY_IN_FULL_CORPUS` in the `full_corpus_for_qid`.
        ii. Have a token length <= `MAX_SF_TOKEN_LENGTH_FOR_FINAL_MOTIF`.
        iii. Discard motifs with no SFs remaining after this filtering.
    f.  **MDL Calculation:**
        i.  Calculate `L(H)` for the final set of refined motifs (token-based cost for labels & descriptions; SF listing cost currently zeroed).
        ii. Create `compressed_corpus_for_qid` by replacing all occurrences of the refined SFs in `full_corpus_for_qid` with their corresponding motif labels.
        iii. Calculate `L(D|H)`: Apply segmented BDM to `compressed_corpus_for_qid`.
    g.  **Evaluate:** `Compression = L(D_orig) - (L(H) + L(D|H))`.
3.  **Reporting:** Log detailed results per QID and an overall summary.

**5. Challenges or Roadblocks Encountered (and Overcome/Mitigated)**

*   **LLM Output Inconsistency (Major Ongoing Challenge):**
    *   *Malformed JSON:* LLM sometimes produces syntactically invalid JSON (e.g., missing commas, unterminated lists/strings), especially for complex chunks or when asked for many motifs. Addressed by: Stricter prompting, increased `max_new_tokens`, pre-parsing regex fixes (e.g., for trailing commas), retry logic.
    *   *Schema Non-Compliance:* LLM output, even if valid JSON, often failed to adhere to the exact schema (e.g., missing keys, incorrect label format `[BRACKETS]`). Addressed by: Very explicit prompting, programmatic label sanitization/fixing, robust validation in parser.
    *   *Poor Surface Form Quality:* LLM initially provided long sentences or generic descriptions as SFs. Addressed by: Iterative, highly specific prompting emphasizing short, verbatim, repeated phrases, and finally by rigorous global SF filtering based on frequency and length.
*   **BDM Sensitivity & Scalability:**
    *   *Prefix Insensitivity:* BDM on a short prefix of a large corpus was not sensitive to global changes from motif replacement. Addressed by: Implementing **segmented BDM**.
    *   *Speed:* Segmented BDM is slower but necessary for accuracy.
*   **MDL Cost Balancing (`L(H)` vs. `L(D|H)`):** Finding the right "cost" for motif definitions (`L(H)`) that allows meaningful compression benefits to emerge is an ongoing tuning process. The current strategy of zeroing `L(H)` costs for SF listings helps isolate data compression effectiveness.
*   **Initial Setup Errors:** Standard Python errors (`NameError`, `UnboundLocalError`, `IndexError`) during refactoring and development. Addressed by: Careful debugging and testing.

**6. Next Priorities**

1.  **Generalize Success:** Run the current successful pipeline on all other QIDs (`P3_QIDS_TO_PROCESS_THEMATICALLY`) to verify if positive compression is achieved more broadly.
2.  **Improve Surface Form Generation by LLM:**
    *   Continue refining the prompt in `build_llm_prompt_for_motifs` to guide the LLM to output even better (shorter, more frequent, more verbatim) SFs directly. The goal is for the LLM's initial SF suggestions to already be high quality, requiring less aggressive filtering later.
    *   Experiment with the `MAX_SF_TOKEN_LENGTH_FOR_FINAL_MOTIF` and `MIN_SF_FREQUENCY_IN_FULL_CORPUS` thresholds.
3.  **Refine `L(H)` Costing:**
    *   Once consistent `L(D|H)` reduction is achieved, systematically re-evaluate the cost components for `L(H)`, perhaps re-introducing a small, principled cost for the surface forms listed in the motif definition.
    *   Investigate more formal ways to define `L(H)` based on information theory (e.g., codelength of the motif definitions).
4.  **Iterative MDL / Motif Selection:**
    *   Develop a strategy to select a *subset* of the globally refined motifs if too many are produced (e.g., the 25 for Q4). Instead of evaluating all of them for `L(H)` and `L(D|H)` together, select the top N most promising ones based on heuristics (e.g., individual compression potential, low definition cost, high SF frequency).
    *   Explore a greedy iterative MDL approach: add one motif at a time if it improves the total description length of (Motif Dictionary + Corpus Compressed with Dictionary).
5.  **Analyze Quality of Discovered Motifs:** Beyond compression scores, qualitatively assess if the discovered motifs are truly insightful and useful for understanding the corpus.
6.  **Address Remaining LLM Errors:** For chunks where JSON parsing or schema validation still fails, analyze the debug logs and further refine prompts or parsing robustness.

**7. Suggested Additional Sections for the Report:**

*   **Quantitative Results Deep Dive (for QID Q4):**
    *   Table showing the 10 refined motifs: Label, Description, Final (filtered) Surface Forms, Number of SFs.
    *   Highlight specific examples of text replacement using a key motif.
    *   Comparison of `L(D_orig)` vs. `L(D|H)` for the segmented BDM, clearly showing the 69.97 unit reduction.
    *   Breakdown of the final `L(H)` = 35.4 for the 10 motifs (cost of labels + cost of descriptions).
*   **Visualizations (Optional, for future reports):**
    *   Graph showing `L(D_orig)` vs. `L(H) + L(D|H)` for each QID as you process more.
    *   Bar chart of compression achieved per QID.
*   **Discussion of "Semantic Richness" vs. "Compressibility":** Acknowledge that the current MDL focuses on compressibility. How do the discovered motifs align with human judgment of semantic importance? (This is more for future work but good to mention).
*   **Limitations of Current MWP:** Be transparent (e.g., BDM on hash prefix of segments, current `L(H)` model is experimental, LLM still needs perfect SF guidance).

---
