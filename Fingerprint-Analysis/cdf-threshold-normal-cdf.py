
import os
import pandas as pd
import numpy as np
from scipy.stats import norm

# Function to process each CSV file
def process_csv(file_path):
    # Load the file
    df = pd.read_csv(file_path)

    # Print the column names to debug
    print(f"Processing file: {file_path}")
    print(f"Columns in the file: {df.columns.tolist()}")

    # Standardize the column names by stripping any extra spaces
    df.columns = df.columns.str.strip()

    # Check for variations in the column name
    if 'Hamming Distance' not in df.columns or 'Number of Images' not in df.columns:
        print(f"Error: Expected columns 'Hamming Distance' and 'Number of Images' not found in {file_path}")
        return

    # 1. Calculate the mean and std of the Hamming distances
    hamming_distances = df['Hamming Distance'].values
    number_of_images = df['Number of Images'].values

    # Estimate the weighted mean and standard deviation
    mean = np.average(hamming_distances, weights=number_of_images)
    variance = np.average((hamming_distances - mean) ** 2, weights=number_of_images)
    std = np.sqrt(variance)

    # 2. Calculate the CDF using the normal distribution for thresholds from 0 to 10
    thresholds = list(range(0, 11))  # 0 to 10
    cdf_probabilities = []

    for threshold in thresholds:
        cdf_value = norm.cdf(threshold, loc=mean, scale=std)
        
        # Format the CDF value in scientific notation for all values
        cdf_probabilities.append(f"{cdf_value:.2e}")

    # 3. Create a DataFrame with thresholds and CDF probabilities
    cdf_df = pd.DataFrame({
        'Threshold': thresholds,
        'CDF Probability': cdf_probabilities
    })

    # Generate output file name based on the input file name
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_Normal_CDF.csv'
    output_file_path = os.path.join(output_directory, output_file_name)

    # Save the result to CSV
    cdf_df.to_csv(output_file_path, index=False)


# Specify the directory containing the CSV files
input_directory = 'sensitivity-output'
output_directory = 'sensitivity-output/tmp2'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each CSV file in the directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_directory, file_name)
        process_csv(file_path)

print("Processing completed!")
