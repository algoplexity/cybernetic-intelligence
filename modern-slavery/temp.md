# The first 5 ABNs from our DataFrame
abns_to_check = df['ABN'].head(5).tolist()

print("Please check the following ABNs on the Australian Business Register (ABR):")
print("Website: https://abr.business.gov.au/")
print("-" * 30)

for abn in abns_to_check:
    print(f"Check ABN: {abn}")
