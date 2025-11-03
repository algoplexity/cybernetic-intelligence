
---

# Zero-Knowledge Developer Handover  
**Modern Slavery Act Compliance Engine**  
*“You’ve never seen this project. In 30 minutes, you’ll own it.”*  
*Date: 4 Nov 2025 | Folder: `ModernSlaveryProject4`*

---

## 1. The 3 Files That Run Everything

| File | What It Is | Never Delete |
|------|------------|--------------|
| `entity_profiles_v2.parquet` | **Every company in Australia** (12.9 million) with **correct name + ABN** | YES |
| `corporate_obligation_log.parquet` | **Who must file** (>$100M revenue, from ATO) | YES |
| `action_log_final_v2.parquet` | **Who actually filed** (from the public register) | YES |

> These 3 files = **your new database**.  
> Everything else is built from them.

---

## 2. How to Open Them (3 Clicks)

```bash
# 1. Install (once)
pip install pandas pyarrow

# 2. Open any file (copy-paste)
import pandas as pd
df = pd.read_parquet("entity_profiles_v2.parquet")
print(df.head())
```

> **No SQL. No cloud. Just Python + Excel.**

---

## 3. The Golden Rule (Never Break This)

> **ABN is God. ACN is trash.**  
> Only match on **ABN**.  
> If no ABN → **do not trust**.

---

## 4. How the Main Report Is Built (Copy-Paste This)

```python
# 1. Load the 3 truth files
entities   = pd.read_parquet("entity_profiles_v2.parquet")
obligated  = pd.read_parquet("corporate_obligation_log.parquet")
statements = pd.read_parquet("action_log_final_v2.parquet")

# 2. Merge: Who should file + who did
report = obligated.merge(statements, on="ABN", how="left")

# 3. Label
report["Status"] = report["Statement_ID"].apply(
    lambda x: "Published" if pd.notna(x) else "Non-Lodger"
)

# 4. Save
report.to_csv("primary_analytical_report.csv", index=False)
```

> **This is the entire engine.**  
> Run this → you have the official report.

---

## 5. How to Add New Data (Next Year)

| New Data | Where to Get It | How to Add |
|--------|------------------|----------|
| **New ATO revenue file** | ATO website (CTT) | Save as CSV → run `scripts/update_obligations.py` |
| **New register statements** | MSA public CSV | Save as CSV → run `scripts/update_actions.py` |
| **New company names** | ABR monthly dump | Save as CSV → run `scripts/update_entities.py` |

> **All scripts are in `/scripts` folder.**  
> Just drop new file → run script → done.

---

## 6. How to Fix a Missing Company (e.g., “Allianz”)

```python
# 1. Search by name (not ABN)
name = "ALLIANZ"
matches = entities[entities["Name"].str.contains(name, case=False)]

# 2. Pick the one with ABN
abn = matches.iloc[0]["ABN"]

# 3. Add to investigative report
print(f"Found ABN: {abn} → add to investigative_report.csv")
```

> **This is how we caught Allianz.**  
> Do this for any “missing” company.

---

## 7. How to Improve It (Your Job Now)

| Want to… | Do This |
|---------|---------|
| **Speed it up** | Add `df = df.astype("category")` for name columns |
| **Add ACN matching** | Only if ABN missing → use `fuzzywuzzy` → log in `abn_acn_exceptions_log.csv` |
| **Build a dashboard** | Load `primary_analytical_report.csv` into Power BI |
| **Automate weekly** | Add to cron: `0 2 * * 1 python rebuild.py` |

---

## 8. The 3 Scripts You Must Run (In Order)

```bash
python scripts/update_entities.py      # 1. Refresh company list
python scripts/update_obligations.py  # 2. Refresh who must file
python scripts/update_actions.py      # 3. Refresh who filed
python rebuild.py                     # 4. Generate new report
```

> **Run every quarter. Takes <10 minutes.**

---

## 9. Emergency Button (If It Breaks)

```bash
# Reset to known good state
git clone https://github.com/your-org/msa-engine.git
cd msa-engine
python rebuild.py
```

> **All code is in GitHub. No secrets. No magic.**

---

## 10. Your First Task (Do This Today)

```bash
# 1. Open terminal
# 2. Run:
python rebuild.py

# 3. Open: primary_analytical_report.csv
# 4. Find 1 Non-Lodger → fix name in entity_profiles_v2.parquet
# 5. Re-run → watch it disappear
```

---
