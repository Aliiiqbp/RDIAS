import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to aggregate Hamming Distance from all CSV files in a directory
def aggregate_hamming_distances(directory):
    all_hamming_distances = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            if 'Hamming Distance' in df.columns:
                all_hamming_distances.extend(df['Hamming Distance'].values)
    return np.array(all_hamming_distances)


# Function to calculate True Positive Rate (TPR) and False Positive Rate (FPR)
def calculate_tpr_fpr(hamming_distances1, hamming_distances2, threshold):
    # TPR: Proportion of true positives (from directory1) correctly classified
    tpr = np.sum(hamming_distances1 <= threshold) / len(hamming_distances1)

    # FPR: Proportion of false positives (from directory2) incorrectly classified as positive
    fpr = np.sum(hamming_distances2 <= threshold) / len(hamming_distances2)

    return tpr, fpr


# Datasets to analyze
Datasets = ["CLIC", "DIV2K"]

# Threshold values from 0 to 10, increasing by 2
thresholds = np.arange(0, 12, 2)

for data in Datasets:
    # Directories containing the CSV files
    directory1 = data + '-verification/CP'
    directory2 = data + '-verification/CC'

    # Aggregate Hamming Distances from both directories
    hamming_distances1 = aggregate_hamming_distances(directory1)
    hamming_distances2 = aggregate_hamming_distances(directory2)

    # Initialize lists to store TPR and FPR for each threshold
    tpr_list = []
    fpr_list = []

    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        tpr, fpr = calculate_tpr_fpr(hamming_distances1, hamming_distances2, threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Plot the ROC curve for the dataset
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, marker='o', linestyle='-', color='b', label=f'ROC Curve ({data})')

    # Plot the random guess line (diagonal)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")

    # Plot settings
    plt.title(f'ROC Curve for {data}', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
