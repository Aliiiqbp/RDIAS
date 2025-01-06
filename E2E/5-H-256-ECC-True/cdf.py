import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Function to aggregate Hamming Distance from all CSV files in a directory
def aggregate_hamming_distances(directory):
    all_hamming_distances = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            if 'Total Hamming Distance' in df.columns:
                all_hamming_distances.extend(df['Total Hamming Distance'].values)
    return np.array(all_hamming_distances)


# Function to calculate and plot the CDF
def plot_cdf(hamming_distances, color, label, threshold):
    sorted_data = np.sort(hamming_distances)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Find the CDF value for the threshold
    cdf_at_threshold = np.sum(sorted_data <= threshold) / len(sorted_data)

    # Plot the CDF
    plt.plot(sorted_data, cdf, color=color, label=f"{label} CDF (Threshold ≤ {threshold})")

    if label == "Directory 2 (Red)":
        print(f"CDF for {label} at threshold {threshold}: {100 * (1 - cdf_at_threshold):.6f}")
    else:
    # Print the CDF at the threshold
        print(f"CDF for {label} at threshold {threshold}: {100 * cdf_at_threshold:.6f}")

    return cdf_at_threshold


Datasets = ["CLIC-F"] # , "DIV2K"

thresholds = [i for i in range(20)]  # i for i in range(0, 11)

for threshold in thresholds:
    for data in Datasets:
        print(data, threshold)    # Directories containing the CSV files
        # Directories containing the CSV files
        directory1 = data + '-verification/CP'
        directory2 = data + '-verification/CC'

        # Aggregate Hamming Distances from both directories
        hamming_distances1 = aggregate_hamming_distances(directory1)
        hamming_distances2 = aggregate_hamming_distances(directory2)


        # Plotting the aggregated Hamming Distance CDF for both directories
        plt.figure(figsize=(12, 8))

        # Directory 1 (Green)
        plot_cdf(hamming_distances1, color='green', label="Directory 1 (Green)", threshold=threshold)

        # Directory 2 (Red)
        plot_cdf(hamming_distances2, color='red', label="Directory 2 (Red)", threshold=threshold)

        # Enhancing the plot aesthetics
        plt.xlim([0, 400])
        plt.title(f"CDF of Hamming Distance (Threshold: {threshold}) - {data}", fontsize=16)
        plt.xlabel("Hamming Distance", fontsize=14)
        plt.ylabel("Cumulative Probability", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # Show the plot
        plt.tight_layout()
        # plt.show()
