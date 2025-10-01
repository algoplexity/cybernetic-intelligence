
***

### **Data Quality and Integrity Report**

This report confirms your initial warning: **none of the columns can be trusted at face value.** The data contains significant structural issues, including inconsistent formats, missing identifiers, unstructured text, and artifacts from previous analyses (e.g., pivot tables).

Below is a file-by-file breakdown of the key findings and the required cleaning actions.

---

#### **File 1: `All time data from Register.xlsx`**

This file is our primary source for submission history but is riddled with formatting and structural problems.

**1. `Statements` Tab - The Core Truth is Hidden in Text**

*   **`Reporting entities` (Critical Column):**
    *   **Finding:** This column is a free-text blob containing company names, ABNs, and other text. The profiling confirms 100% of non-null values are complex strings ("other").
    *   **Challenge:** This is our *only* source linking a submitted statement (`Statement #`) to the entity's ABN. A single entry can contain multiple ABNs for joint statements.
    *   **Action Plan:**
        1.  Implement a robust regular expression to find and extract all valid 11-digit ABNs from each cell.
        2.  Create a separate "linking table" (`Statement ID` to `ABN`) to correctly handle one-to-many relationships for joint statements.

*   **`Revenue`:**
    *   **Finding:** A text field with ranges ("100-150M"), absolutes ("1BN+"), and text ("Unknown").
    *   **Action Plan:** Parse this column to create two new, clean columns: a `Revenue Category` (text) and a `Revenue Minimum` (numeric), which will allow for proper filtering (e.g., `> 100,000,000`).

*   **Date Columns (`Submitted`, `Date published`, `Period start/end`):**
    *   **Finding:** These are all text fields representing dates, not actual date objects. The format appears to be `YYYY-MM-DD HH:MM:SS`.
    *   **Action Plan:** Parse all date-related columns into a standardized `YYYY-MM-DD` date format, handling any potential errors gracefully.

*   **Flag & Industry Columns (`16(1)(a) `, `Signature`, `Financial...` etc.):**
    *   **Finding:** The column headers have **trailing spaces** (e.g., `'16(1)(a) '` instead of `'16(1)(a)'`). The data itself is consistently '0' or '1' but stored as text.
    *   **Action Plan:**
        1.  **Crucially, strip all leading/trailing whitespace from every column header upon loading.**
        2.  Convert these columns to a numeric (integer) or boolean data type for proper analysis.

**2. `Entities` Tab - Unreliable for Primary Identification**

*   **`ABN` (Critical Finding):**
    *   **Finding:** **34.2% of ABNs in this tab are NULL.**
    *   **Challenge:** This makes the tab completely unreliable as a master list of entities. We cannot use it as our primary source of ABNs.
    *   **Action Plan:** This tab will be used as a *secondary* source for cross-referencing company names only. The primary ABN for a given statement *must* come from the `Statements` tab.

*   **`Company name`:**
    *   **Finding:** Contains a mix of cases, suffixes ("PTY LTD"), and special characters.
    *   **Action Plan:** Standardize all names to uppercase, trim whitespace, and normalize common suffixes to ensure consistency with other files.

*   **`Reporting years` & `Statements submitted`:**
    *   **Finding:** These are concatenated text blobs, effectively useless for direct querying.
    *   **Action Plan:** Ignore these columns. We will generate our own, accurate submission summaries from the `Statements` data.

**3. `Holiday`, `LK`, `DASH`, `Annual Report` Tabs - To Be Excluded**

*   **Finding:** The profiling confirms these are not raw data. They contain pivot tables, pre-calculated summaries, and many unnamed columns.
*   **Action Plan:** These tabs will be **explicitly ignored** during the creation of our clean data mart. They are irrelevant to our core task.

---

#### **File 2: `ato_tax_transparency_non_lodger.xlsx`**

This file will serve as our "universe" of potential reporters. Its data quality is generally higher but has format quirks.

*   **`Non-Lodger` Tab:**
    *   **Finding:** `ABN` column is 100% populated and numeric. This is excellent and will be the **primary key for our Master Entity File.**
    *   **Finding:** Date columns (`Abn_regn_dt`, `GST_regn_dt`) are stored as numbers (e.g., `20091020`).
    *   **Action Plan:** Convert these numeric date columns to the standard `YYYY-MM-DD` date format.
    *   **Finding:** Many columns are entirely null (e.g., `prty_id_blnk`, `nm_sufx_cd`).
    *   **Action Plan:** These empty columns will be dropped to keep the master file clean and focused.

*   **`Associates` Tab:**
    *   **Finding:** Data is generally clean and well-structured. Null values for names are expected.
    *   **Action Plan:** Standardize name fields (trimming whitespace, standardizing case) to create our `clean_associates.csv` file for deep-dive analysis.

---

#### **Files 3 & 4: `lodge_once.csv` & `lodge_once_cont.xlsx`**

These files contain critical information about single-lodgers but have a major identifier problem.

*   **`abn` (Critical Finding):**
    *   **Finding:** The profiling reveals that **21.6% of ABNs are dummy/masked values** (e.g., "dummy\_2918"). The remaining 78.4% appear to be real ABNs.
    *   **Challenge:** We cannot join the records with dummy ABNs to our master file.
    *   **Action Plan:**
        1.  The two `lodge_once` files will be merged into one.
        2.  The data will be split into two groups: records with **valid ABNs** and records with **dummy ABNs**.
        3.  Only the data with valid ABNs will be joined to the Master Entity File. The "dummy" data will be saved to a separate file for potential future investigation, should a key be found.

*   **`last_submission_dttm`:**
    *   **Finding:** Dates are in a specific ISO format (`YYYY-MM-DDTHH:MM:SSZ`).
    *   **Action Plan:** Use a specific parser to handle this format correctly.

***

### **Conclusion: The Defensive Cleaning Strategy**

This detailed profile gives us a clear "battle plan." The subsequent cleaning script will be built defensively to handle these specific, now-documented issues. Every step will be justified by the findings in this report.

We have a solid plan to navigate the data's poor quality and produce the reliable, two-component **Clean Data Mart** we need for the investigation.

