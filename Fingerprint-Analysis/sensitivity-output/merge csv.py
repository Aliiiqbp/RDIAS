import os
import pandas as pd


# Function to merge all CSV files in the directory
def merge_cdf_files(input_directory, output_file_path):
    merged_df = None

    # Process each CSV file in the directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_directory, file_name)

            # Load the current CSV file
            df = pd.read_csv(file_path)

            # Get the file name without the extension for the column name
            column_name = os.path.splitext(file_name)[0]

            # Rename the 'CDF Probability' column to the file name
            df = df.rename(columns={'CDF Probability': column_name})

            # Merge the data on the 'Threshold' column
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Threshold', how='outer')

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file_path, index=False)


# Specify the directory containing the CSV files
input_directory = 'tmp2'
output_file_path = 'tmp2/merged_CDF_results.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Merge the CDF files
merge_cdf_files(input_directory, output_file_path)

print("Merging completed!")
