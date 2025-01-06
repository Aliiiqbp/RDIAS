import pandas as pd

# Load the finalized CSV file
input_file = "byte_index_file.csv"  # Replace with your finalized CSV file name
df = pd.read_csv(input_file)

# Filter rows where byte_index is not zero
filtered_df = df[df['byte_index'] != 0]

# Group by the "Transformation" column and calculate the average byte_index
average_byte_index = filtered_df.groupby('Transformation')['byte_index'].mean()

# Print the results
print("Average byte_index (ignoring zeros) for each Transformation:")
print(average_byte_index)
