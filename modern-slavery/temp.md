--- Deep Dive Analysis Results ---

[Analysis 1: GST Status of Financial Sector Non-Lodgers]
GST_Status
ACT    263
NON     98
CAN     14
Name: count, dtype: int64

Insight: 'CAN' indicates a cancelled GST registration, a strong sign of a dormant or non-trading entity.

[Analysis 2: Age Distribution of Financial Sector Non-Lodgers (in years)]
count    375.0
mean      13.8
std        7.9
min        0.1
25%        6.8
50%       11.8
75%       21.8
max       25.9
Name: CompanyAge, dtype: float64

Insight: This tells us if these are new or long-established entities.
/tmp/ipython-input-521476279.py:45: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  financial_sector_df['CompanyAge'] = (datetime.now() - financial_sector_df['RegistrationDate']).dt.days / 365.25
