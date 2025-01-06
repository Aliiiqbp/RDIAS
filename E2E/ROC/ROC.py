import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Custom transformation to map y-axis values to equal visual spacing
def custom_y_transform(y):
    # Apply a nonlinear transformation to "stretch" the y-values to equal visual distance
    return np.log10(y - 0.49)  # Shifts to make 50% non-zero and usable in log scale

# List of experiment filenames
experiment_files = ['CLIC_dataset_1.csv', 'CLIC_dataset_2.csv', 'CLIC_dataset_3.csv', 'CLIC_dataset_4.csv']

# Initialize the plot
plt.figure(figsize=(8, 6))

for i, file in enumerate(experiment_files):
    # Read the data
    data = pd.read_csv(file)
    # Replace 0 FPR with a small value to avoid log scale issues
    data['FP'] = np.where(data['FP'] == 0, 1e-6, data['FP'])
    # Filter data to only include TPR values above 50% (0.5)
    data_filtered = data[data['TP'] > 0.5]
    # Apply custom transformation for y-axis values
    plt.plot(data_filtered['FP'], custom_y_transform(data_filtered['TP']), marker='o', label=f'Experiment {i+1}')

# Generate x values spaced logarithmically between 1e-6 and 1 for random guess
x_values = np.logspace(-6, 0, 100)
# For random guessing, y = x, but apply custom transformation
x_values_filtered = x_values[x_values > 0.5]
plt.plot(x_values_filtered, custom_y_transform(x_values_filtered), 'k--', label='Random Guess')

# Set x-axis to log scale (base 10)
plt.xscale('log')

# Customize x-ticks to show 1, 0.1, 0.01, etc.
plt.xticks([1, 0.1, 0.01, 0.001, 0.0001, 1e-6], ['1', '0.1', '0.01', '0.001', '0.0001', '1e-6'])

# Set custom y-ticks with equal visual spacing
y_tick_values = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
y_tick_labels = ['50%', '90%', '99%', '99.9%', '99.99%', '99.999%']
plt.yticks(custom_y_transform(np.array(y_tick_values)), y_tick_labels)

plt.xlabel('False Positive Rate (Log Scale)')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Experiments (TPR > 50%)')
plt.legend()
plt.grid(True)
plt.show()
