
---

### **Walkthrough Script: Month 1 Analysis Deliverable**

**Objective:** To present the initial findings from the Month 1 analysis, explain the structure and significance of the curated dataset, and gather feedback for the next phase.

**(1) Introduction & Setting the Scene)**

"Good morning/afternoon, [Stakeholder Name]. Thanks for making the time.

The purpose of this brief walkthrough is to share the initial deliverable from the first phase of the Modern Slavery Act data analysis project.

As we discussed, the primary goal for Month 1 was to complete the **Entity Lodgement Analysis**—essentially, to sift through all the available data and produce a clean, foundational dataset of key entity cohorts.

The Excel file I've sent over is that deliverable. It provides the first clear, data-driven shape of the compliance landscape. I'd like to walk you through the two main tabs and then discuss how this sets us up for the deeper analysis in Month 2."

---

**(2) File Overview - Navigating the Excel Sheet)**

"If you open the `Month_1_Analysis_Deliverable_Automated.xlsx` file, you'll see two tabs at the bottom:
1.  **`Never Lodged Entities`**
2.  **`Single Lodgement Entities`**

These two tabs represent the two critical cohorts we identified as per the project plan. I'll start with the first one."

---

**(3) Deep Dive: Tab 1 - `Never Lodged Entities`)**

"This first sheet, **`Never Lodged Entities`**, contains the list of **1,343** high-revenue companies that we believe may have never submitted a Modern Slavery Statement.

**How we got this list:** We took the list of companies from the ATO's tax transparency data—all with revenues over the reporting threshold—and cross-referenced it against the master Register of every statement ever submitted. These 1,343 entities are the ones that appeared on the ATO list but not in our Register.

**What you're seeing in the columns:**
*   You'll see standard identifiers like **`ABN`** and **`Entity Name`**.
*   Columns like **`Total Income`** and **`Entity size`** provide the evidence for why they are likely required to report.
*   We've also included other available ABR details like **`Industry_desc`**, **`State`**, and contact details like **`Ent_eml`** where they were available.

**Why this is significant:** This list represents our first, targeted view of a high-risk group. It gives us a clear set of entities that may require follow-up or be prioritized for compliance attention."

---

**(4) Deep Dive: Tab 2 - `Single Lodgement Entities`)**

"Now, if you click over to the second tab, **`Single Lodgement Entities`**, this is the second key cohort. This list contains the **4,198** entities that have submitted a statement exactly once.

**How we got this list:** This list was generated directly from our master Register. We performed a frequency count to find every single entity that has only appeared once, so this is a definitive list.

**What you're seeing in the columns (This is important):**
*   You'll see columns from the Register itself, like **`Reporting entities`** (the original, sometimes messy, text) and the **`Reporting Period`**.
*   You'll also see columns we've enriched it with, like **`abn`**, **`company_name`**, and crucially, the compliance columns like **`last_submission_dttm`** and **`num_compliant`**.

**A key finding I want to highlight:** You'll notice that for many rows, the enriched columns are blank. This is an important discovery. It tells us that our supplementary data files—the ones with the detailed compliance flags—only contained data for a *subset* of this cohort (about 2,300 of the 4,198 entities).
*   Where you **see data** in columns like `num_compliant`, that entity has been successfully enriched.
*   Where it is **blank**, it means the entity exists on the Register but wasn't in our detailed supplementary files.

**Why this is significant:** This cohort is our primary focus for the Month 2 **Compliance Pattern Analysis**. These are entities that have engaged with the system once, and we need to understand their ongoing compliance behaviour. The enriched data gives us a powerful starting point to analyze exactly which parts of the Act they are finding difficult to comply with."

---

**(5) Conclusion & Next Steps (Inviting Feedback))**

"So, in summary, this Excel file provides a clean, validated, and segmented view of our two initial cohorts of interest: a high-risk group that may have never reported, and a second group that has reported once and warrants deeper analysis.

This work completes the main deliverable for Month 1 and directly prepares us for the **Compliance Pattern Analysis** in Month 2. Our next step will be to analyze the `Single Lodgement Entities` to identify trends in non-compliance with Section 16.

Before we move on, I wanted to open the floor to you. In line with my goal to **Collaborate with Purpose**, your feedback is critical. Does this initial segmentation make sense? Are there any other high-level views or questions that come to mind as you look at this data? This will help ensure our deeper analysis is as useful as possible for the policy team."
