import pandas as pd


def process_csv(file_path, output_path):
    # Step 1: Read the CSV file
    df = pd.read_csv(file_path)

    # Step 2: Group by 'col1' and 'col2'
    grouped = df.groupby(['Transformation', 'Parameter'])

    # Step 3: Calculate the average of non-null values for columns 'col3' and 'col4'
    averages = grouped[['Bit Accuracy', 'burst_index']].mean(numeric_only=True)

    # Step 4: Calculate the sum for columns 'col5' and 'col6'
    sums = grouped[['0-to-1', '1-to-0']].sum(numeric_only=True)

    # Combine the results
    result = averages.join(sums).reset_index()

    # Print the final result
    result.to_csv(output_path, index=False)
    print(result)


# Example usage
file_path = 'Amin_test_watermarking_results.csv'  # Update with your CSV file path
output_path = 'Amin-average.csv'      # Update with your desired output file path
process_csv(file_path, output_path)
