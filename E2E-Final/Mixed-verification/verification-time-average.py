import pandas as pd

# Step 1: Load the two CSV files
file_path1 = 'attack.csv'   # Update with your first CSV file path
file_path2 = 'common.csv'  # Update with your second CSV file path

# Read the CSV files into pandas DataFrames
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# Step 2: Append the two datasets together
combined_data = pd.concat([data1, data2], ignore_index=True)

# Step 3: Calculate the average values for the specified columns
columns_of_interest = ['E2E_time', 'Fingerprinting_time', 'wm_time', 'ECC_time']
averages = combined_data[columns_of_interest].mean()

# Print the average values
print("Average Values:")
print(averages)
