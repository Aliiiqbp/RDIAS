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
            if 'Hamming Distance' in df.columns:
                all_hamming_distances.extend(df['Hamming Distance'].values)
    return np.array(all_hamming_distances)

# Directories containing the CSV files
directory1 = 'CLIC-verification/CP'
directory2 = 'CLIC-verification/CC'

# Aggregate Hamming Distances from both directories
hamming_distances1 = aggregate_hamming_distances(directory1)
hamming_distances2 = aggregate_hamming_distances(directory2)

# Plotting the aggregated Hamming Distance distribution for both directories
plt.figure(figsize=(12, 8))

# Directory 1 (Green)
n1, bins1, patches1 = plt.hist(hamming_distances1, bins=30, range=(0, 400), density=True, alpha=0.5, color='green', edgecolor='black', label="Directory 1 (Green) Data")
mean1, std1 = np.mean(hamming_distances1), np.std(hamming_distances1)
x1 = np.linspace(0, 400, 100)
p1 = stats.norm.pdf(x1, mean1, std1)
plt.plot(x1, p1, 'g-', linewidth=2, label=f'Normal Distribution (Green)\nMean: {mean1:.2f}, Std: {std1:.2f}')

# Directory 2 (Red)
n2, bins2, patches2 = plt.hist(hamming_distances2, bins=30, range=(0, 400), density=True, alpha=0.5, color='red', edgecolor='black', label="Directory 2 (Red) Data")
mean2, std2 = np.mean(hamming_distances2), np.std(hamming_distances2)
x2 = np.linspace(0, 400, 100)
p2 = stats.norm.pdf(x2, mean2, std2)
plt.plot(x2, p2, 'r-', linewidth=2, label=f'Normal Distribution (Red)\nMean: {mean2:.2f}, Std: {std2:.2f}')

# Enhancing the plot aesthetics
plt.xlim([0, 400])
plt.title("Aggregated Hamming Distance Distribution for Two Directories", fontsize=16)
plt.xlabel("Hamming Distance", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
