
import os
import pandas as pd

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

    # 1. Calculate the distribution of Hamming distances
    distribution = df.groupby('Hamming Distance')['Number of Images'].sum()

    # 2. Calculate the CDF for thresholds from 0 to 10
    thresholds = list(range(0, 11))  # 0 to 10
    cdf_probabilities = []

    total_images = df['Number of Images'].sum()

    for threshold in thresholds:
        images_up_to_threshold = df[df['Hamming Distance'] <= threshold]['Number of Images'].sum()
        cdf_value = images_up_to_threshold / total_images
        
        # Format the CDF value in scientific notation for all values
        cdf_probabilities.append(f"{cdf_value:.2e}")

    # 3. Create a DataFrame with thresholds and CDF probabilities
    cdf_df = pd.DataFrame({
        'Threshold': thresholds,
        'CDF Probability': cdf_probabilities
    })

    # Generate output file name based on the input file name
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_CDF.csv'
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
