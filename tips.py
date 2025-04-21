import seaborn as sns
import pandas as pd

# Load the 'tips' dataset
tips = sns.load_dataset('tips')

# Save it to a CSV file
tips.to_csv('tips.csv', index=False)

print("✅ tips.csv has been created.")