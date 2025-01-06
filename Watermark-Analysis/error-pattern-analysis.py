import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'CLIC-686_test_watermarking_results.csv'  # Update with your actual CSV file path
df = pd.read_csv(csv_file_path)

# Initialize variables to store total sum of gaps and total number of gaps
total_gap_sum = 0
total_gap_count = 0
all_gaps = []  # To store all gaps for calculating average and median
gap_distribution = Counter()  # To store the frequency of each gap

# Loop through each row in the dataset
for idx, row in df.iterrows():
    error_indexes = row["Error Indexes"]

    # Check if the error indexes column is non-empty
    if pd.notna(error_indexes) and error_indexes != "":
        # Convert the string of error indexes to a list of integers
        error_positions = list(map(int, error_indexes.split(',')))

        # Only consider rows with more than 1 error to calculate gaps
        if len(error_positions) > 1:
            # Calculate the gaps between consecutive errors
            gaps = np.diff(error_positions)

            # Sum the gaps for this row
            total_gap_sum += np.sum(gaps)

            # Count the number of gaps (len(error_positions) - 1)
            total_gap_count += len(gaps)

            # Store gaps for overall calculations
            all_gaps.extend(gaps)

            # Update the gap distribution
            gap_distribution.update(gaps)

# Calculate the overall average and median gap if there are gaps to consider
if total_gap_count > 0:
    average_gap = total_gap_sum / total_gap_count
    median_gap = np.median(all_gaps)

    print(f"Overall Average Gap Between Consecutive Errors: {average_gap:.2f}")
    print(f"Overall Median Gap Between Consecutive Errors: {median_gap:.2f}")

    # Plot the gap distribution
    gap_values = list(gap_distribution.keys())
    gap_frequencies = list(gap_distribution.values())

    plt.bar(gap_values, gap_frequencies, color='skyblue', edgecolor='black')
    plt.xlabel('Gap Between Errors')
    plt.ylabel('Frequency')
    plt.title('Gap Distribution Between Consecutive Errors')

    # Annotate the plot with average and median lines
    plt.axvline(x=average_gap, color='red', linestyle='--', label=f'Average: {average_gap:.2f}')
    plt.axvline(x=median_gap, color='green', linestyle='--', label=f'Median: {median_gap:.2f}')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()

else:
    print("No gaps to calculate.")
