import numpy as np

print("\n--- Starting Month 2 Analysis: Compliance by Industry Sector ---")

# --- 1. Prepare the Data ---
# The df_with_compliance_data DataFrame is already in memory and contains the 737 entities.
# We will use the 'industry_desc' column for grouping. Let's handle any missing industry descriptions.
df_with_compliance_data['industry_desc'].fillna('Unknown', inplace=True)
print("Step 1/4: Data prepared for industry analysis.")

# --- 2. Calculate Non-Compliance Rate per Industry ---
# We will group by industry and calculate the average non-compliance rate.
# We will also count the number of entities in each industry to ensure our analysis is meaningful.

# First, create a total non-compliance score for each entity.
# The 'num_non_compliant' column should already represent this.
# If it doesn't exist, we can calculate it.
if 'num_non_compliant' not in df_with_compliance_data.columns:
    df_with_compliance_data['num_non_compliant'] = df_with_compliance_data[existing_criteria_cols].sum(axis=1)

industry_summary = df_with_compliance_data.groupby('industry_desc').agg(
    entity_count=('abn', 'count'),
    total_non_compliance=('num_non_compliant', 'sum')
).reset_index()

# Calculate the average number of non-compliant criteria per entity in each industry
industry_summary['avg_non_compliant_criteria'] = industry_summary['total_non_compliance'] / industry_summary['entity_count']
print("Step 2/4: Calculated average non-compliance per industry.")

# --- 3. Identify High-Risk Industries ---
# We will filter for industries with a meaningful number of entities (e.g., 5 or more)
# Then sort them to find those with the highest average non-compliance.
meaningful_industries = industry_summary[industry_summary['entity_count'] >= 5]
top_10_high_risk_industries = meaningful_industries.sort_values(by='avg_non_compliant_criteria', ascending=False).head(10)
print("Step 3/4: Identified top 10 industries with highest average non-compliance.")

# --- 4. Report the Findings ---
print("Step 4/4: Generating final summary report.")
print("\n--- Month 2: Top 10 High-Risk Industries (by Avg. Non-Compliant Criteria) ---")
print("Analysis based on 737 single-lodger entities. Showing industries with 5 or more entities.")
print("--------------------------------------------------------------------------------")
# Set pandas to display the full industry name
pd.set_option('display.max_colwidth', None)
print(top_10_high_risk_industries[['industry_desc', 'entity_count', 'avg_non_compliant_criteria']])
print("--------------------------------------------------------------------------------")
print("--- Analysis Complete ---")
