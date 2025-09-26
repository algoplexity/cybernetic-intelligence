import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

print("--- Starting: Generating Power BI Data Source from Annual Report Metrics ---")

# ==============================================================================
# PART 1: LOAD AND PREPARE ALL RAW DATA SOURCES
# ==============================================================================
try:
    # Load all necessary dataframes from the source files
    df_register = pd.read_excel('All time data from Register.xlsx', sheet_name='Statements')
    df_non_lodger_src = pd.read_excel('ato_tax_transparency_non_lodger.xlsx', sheet_name='Non-Lodger')
    df_lodge_once_comp = pd.read_csv('lodge_once.csv')
    df_lodge_once_details = pd.read_excel('lodge_once_cont.xlsx', sheet_name='lodge_once')

    # --- Perform initial cleaning and preparation ---
    df_register.columns = df_register.columns.str.strip().str.replace('\n', '')
    df_register['Period end date'] = pd.to_datetime(df_register['Period end date'], errors='coerce')
    df_register['Submitted'] = pd.to_datetime(df_register['Submitted'], errors='coerce')
    
    # Create Reporting Cycle column for grouping (approximated from period end date)
    def assign_cycle(date):
        if pd.isna(date): return None
        if date.year == 2021 and date.month <= 6: return 'Reporting Cycle 1'
        if date.year == 2022 and date.month <= 6: return 'Reporting Cycle 2'
        if date.year == 2023 and date.month <= 6: return 'Reporting Cycle 3'
        if date.year == 2023 and date.month > 6: return 'Reporting Cycle 4 (6mo)'
        return 'Other'
    df_register['Reporting Cycle'] = df_register['Period end date'].apply(assign_cycle)

    print("Step 1/4: All source files loaded and prepared successfully.")

except Exception as e:
    print(f"ERROR in Data Loading: {e}")
    raise

# ==============================================================================
# PART 2: CALCULATE METRICS AND CHART DATA FROM RAW SOURCES
# ==============================================================================

# --- Tab 1: Key Metrics (Mixture of Calculated and Transcribed) ---
key_metrics_data = {
    'Metric Description': [
        'Total Searches on Register (by end of 2023)', 'Searches on Register (in 2023 calendar year)',
        'Total Reporting Entities Covered (estimated)', 'Statements Assessed (in 2023 calendar year)',
        'Statements Published (in 2023 calendar year)', 'Reporting Entities Headquarters (countries)',
        'Helpdesk Responses (in 2023)', 'Publishable Statements Published within 60 Days (2023)'
    ],
    'Value': [
        3300000, 1500000, 9500, 3400, 3000, 59, 2200, '99.6%'
    ],
    'Data Source in Report': [
        'Page 11', 'Page 4', 'Page 11', 'Page 9', 'Page 7', 'Page 11', 'Page 33', 'Page 34'
    ]
}
df_key_metrics = pd.DataFrame(key_metrics_data)

# --- Tab 2: Fig 2 - Monthly Submissions (Calculated) ---
df_register['Submission Month'] = df_register['Submitted'].dt.to_period('M')
df_monthly_submissions = df_register.groupby('Submission Month').size().reset_index(name='Number of Submissions')
df_monthly_submissions['Submission Month'] = df_monthly_submissions['Submission Month'].astype(str)

# --- Tab 3 & 4: Industry & Revenue (Calculated) ---
# Note: This is an approximation as entities can select multiple industries.
industry_cols = ['Financial, insurance and real estate activities', 'Construction, civil engineering and building products', 'Food and beverages, agriculture and fishing', 'Mining, metals, chemicals and resources (including oil and gas)', 'Information technology and telecommunication', 'Healthcare and pharmaceuticals', 'Transportation, logistics, and storage']
df_industry_long = df_register.melt(id_vars=['Reporting Cycle'], value_vars=industry_cols, var_name='Industry Sector', value_name='Is_Selected')
df_industry_long = df_industry_long[df_industry_long['Is_Selected'] == 1]
df_industry_summary = pd.crosstab(df_industry_long['Industry Sector'], df_industry_long['Reporting Cycle'], normalize='columns').apply(lambda x: x * 100).reset_index()

df_revenue_summary = pd.crosstab(df_register['Revenue'], df_register['Reporting Cycle'], normalize='columns').apply(lambda x: x * 100).reset_index()

print("Step 2/4: Metrics from raw data calculated successfully.")

# ==============================================================================
# PART 3: TRANSCRIBE DATA NOT AVAILABLE IN RAW FILES
# ==============================================================================

# --- Tab 5: Table 2 & Other Compliance Trends (Transcribed) ---
compliance_trends_data = {
    'Metric': [
        'Number of Statements Submitted', '% Assessed as Publishable', '% Assessed as Non-Publishable',
        '% Publishable Stmts Assessed as Compliant', '% Publishable Stmts Assessed as Non-Compliant',
        '% Voluntary Statements of Total', '% with Overseas Obligations'
    ],
    'Reporting Cycle 1': [2300, '73%', '27%', '59%', '41%', '4.0%', '20.0%'],
    'Reporting Cycle 2': [3200, '83%', '17%', '71%', '29%', '5.0%', '21.0%'],
    'Reporting Cycle 3': [3350, '88%', '12%', '74%', '26%', '7.0%', '21.5%'],
    'Reporting Cycle 4 (6mo)': [1700, '89%', '11%', '85%', '15%', '7.8%', '38.0%']
}
df_compliance_trends = pd.DataFrame(compliance_trends_data)

# --- Tab 6: Fig 9 & 10 - Non-Publishable Deep Dive (Transcribed) ---
non_publishable_data = {
    'Metric': [
        'Reason: Signature from Responsible Member', 'Reason: Principal Governing Body Approval', 'Reason: Both Missing',
        'Re-submission Rate: % Re-submitted', 'Re-submission Rate: % Not Re-submitted'
    ],
    'Reporting Cycle 1': ['9.0%', '78.0%', '12.0%', '93%', '7%'],
    'Reporting Cycle 2': ['11.0%', '79.0%', '15.0%', '95%', '5%'],
    'Reporting Cycle 3': ['25.0%', '55.0%', '20.0%', '91%', '9%']
}
df_non_publishable = pd.DataFrame(non_publishable_data)

# --- Tab 7: Fig 12-15 - Mandatory Criteria Deep Dive (Transcribed) ---
criteria_data = {
    'Mandatory Criteria Failure / Trend': [
        'Non-Compliance Rate: (f) Consultation Process', 'Non-Compliance Rate: (e) Effectiveness of Actions',
        'Non-Compliance Rate: (d) Actions Taken', 'Non-Compliance Rate: (c) Risks of Modern Slavery',
        'Non-Compliance Rate: (b) Structure and Supply Chains', 'Non-Compliance Rate: (a) Identify Reporting Entity',
        '% Compliant with 16(1)(c)', '% Compliant with 16(1)(e)', '% Compliant with 16(1)(f)'
    ],
    'Reporting Period 1': ['58%', '28%', '5%', '2%', '2%', '1%', '97%', '72%', '42%'],
    'Reporting Period 2': ['52%', '28%', '7%', '4%', '3%', '1%', '90%', '75%', '48%'],
    'Reporting Period 3': ['45%', '25%', '3%', '19%', '2%', '0.5%', '80%', '78%', '55%'],
    'Reporting Period 4 (6mo)': ['55%', '30%', '5%', '30%', '5%', '0.5%', '70%', '79%', '45%']
}
df_criteria = pd.DataFrame(criteria_data)

print("Step 3/4: Data transcribed from report visuals successfully.")

# ==============================================================================
# PART 4: WRITE ALL DATAFRAMES TO A SINGLE EXCEL FILE
# ==============================================================================
output_filename = 'Annual_Report_Metrics_for_PowerBI.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_key_metrics.to_excel(writer, sheet_name='Key Metrics', index=False)
    df_monthly_submissions.to_excel(writer, sheet_name='Fig 2 - Monthly Submissions', index=False)
    df_industry_summary.to_excel(writer, sheet_name='Fig 3 - Industry Sectors', index=False)
    df_revenue_summary.to_excel(writer, sheet_name='Fig 4 - Annual Revenue', index=False)
    df_compliance_trends.to_excel(writer, sheet_name='Compliance Trends (T2, F5-8,11)', index=False)
    df_non_publishable.to_excel(writer, sheet_name='Non-Publishable (Fig 9,10)', index=False)
    df_criteria.to_excel(writer, sheet_name='Mandatory Criteria (Fig 12-15)', index=False)

print(f"Step 4/4: All data successfully written to '{output_filename}'.")
print("\n--- Pipeline Complete ---")
