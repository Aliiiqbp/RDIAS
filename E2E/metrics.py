import pandas as pd

# Replace 'tpr_file.csv' and 'fpr_file.csv' with your actual file names
tpr_file = 'TPR.csv'  # File containing Threshold and TPR columns
fpr_file = 'FPR.csv'  # File containing Threshold and FPR columns
output_file = 'metrics.csv'  # Output file to save the computed metrics

# Read the TPR file
tpr_df = pd.read_csv('TPR-90.csv')

# Read the FPR file
fpr_df = pd.read_csv('FPR-90.csv')

# Merge the two DataFrames on 'Threshold'
df = pd.merge(tpr_df, fpr_df, on='Threshold')

# Compute the metrics
df['ACC'] = (df['TPR'] + 1 - df['FPR']) / 2
df['Precision'] = df['TPR'] / (df['TPR'] + df['FPR'])
df['Recall'] = df['TPR']
df['F1-score'] = 2 * df['Precision'] * df['Recall'] / (df['Precision'] + df['Recall'])

# Handle possible division by zero in Precision and F1-score calculations
df['Precision'] = df['Precision'].fillna(0)
df['F1-score'] = df['F1-score'].fillna(0)

# Save the computed metrics to a new CSV file
df.to_csv(output_file, index=False)

print(f"Metrics have been computed and saved to '{output_file}'.")
