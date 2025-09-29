import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
sns.set_style("whitegrid")

print("--- Final Analysis: Profiling Non-Lodgers by Entity Type ---")

# --- Analysis 1: Overall Distribution of Entity Types ---
print("\n[Analysis 1: Most Common Entity Types (All 1,338 Non-Lodgers)]")
top_10_entity_types = potential_reporters['EntityType'].value_counts().nlargest(10)
print(top_10_entity_types)

# Plotting the overall distribution
plt.figure(figsize=(10, 7))
sns.barplot(y=top_10_entity_types.index, x=top_10_entity_types.values, palette="plasma")
plt.title('Top 10 Entity Types for All Non-Lodgers', fontsize=16)
plt.xlabel('Number of Entities', fontsize=12)
plt.ylabel('Entity Type', fontsize=12)
plt.show()


# --- Analysis 2: Deep Dive into the "Financial Asset Investing" Sector ---
print("\n[Analysis 2: Deep Dive on 'Financial Asset Investing' Sector]")
# Filter our data for just this industry
financial_sector_df = potential_reporters[potential_reporters['Industry'] == 'Financial Asset Investing']
financial_entity_types = financial_sector_df['EntityType'].value_counts().nlargest(10)

print(f"Breakdown of the {len(financial_sector_df)} entities in Financial Asset Investing:")
print(financial_entity_types)

# Plotting the financial sector distribution
plt.figure(figsize=(10, 7))
sns.barplot(y=financial_entity_types.index, x=financial_entity_types.values, palette="cividis")
plt.title('Top Entity Types within "Financial Asset Investing"', fontsize=16)
plt.xlabel('Number of Entities', fontsize=12)
plt.ylabel('Entity Type', fontsize=12)
plt.show()
