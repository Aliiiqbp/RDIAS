import os
import pandas as pd
from coremltools.converters.mil.mil.ops.defs.iOS15 import threshold

# Define the directory containing your CSV files and set your threshold value
directory = '../Mixed-verification'  # Replace with your actual directory

for threshold in [i for i in range(11)]:
    # threshold = 5  # Set your threshold value here

    # Initialize an empty list to store the results
    results = []

    # Iterate through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            # Load the CSV file
            df = pd.read_csv(file_path)

            # Check if the required column exists
            if "Total Hamming Distance" in df.columns:
                # Count rows based on the threshold
                less_equal_count = (df["Total Hamming Distance"] <= threshold).sum()
                greater_count = (df["Total Hamming Distance"] > threshold).sum()

                # Append the results to the list
                results.append({
                    'Filename': filename,
                    'Less or Equal to Threshold': less_equal_count,
                    'Greater than Threshold': greater_count
                })

    # Convert results to a DataFrame and save to a CSV file
    output_df = pd.DataFrame(results)
    output_file = 'plots/results_summary_' + str(threshold) + '.csv'
    output_df.to_csv(output_file, index=False)

    # Print a message to indicate completion
    print(f"Results have been saved to {output_file}")
