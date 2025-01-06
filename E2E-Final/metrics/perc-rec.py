import pandas as pd
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

# Load the CSV file
file_path = 'metrics.csv'
data = pd.read_csv(file_path)


# Function to filter data by threshold and plot Precision vs. Recall
def plot_precision_recall_by_threshold(data, threshold_min, threshold_max):
    # Filter the data by the specified threshold range
    filtered_data = data[(data['Threshold'] >= threshold_min) & (data['Threshold'] <= threshold_max)]

    if filtered_data.empty:
        print(f"No data available in the threshold range {threshold_min} to {threshold_max}.")
        return

    # Ensure the required columns are present
    if 'Precision' in filtered_data.columns and 'Recall' in filtered_data.columns:
        # Plot Precision vs. Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(filtered_data['Recall'], filtered_data['Precision'], marker='', linestyle='-',
                 label='RDIAS', color='blue', linewidth=4)
        plt.plot([100, 50], [50, 50], 'r--', label='Random Classifier')
        plt.xlabel('Recall (%)', fontsize=42)
        plt.ylabel('Precision (%)', fontsize=42)
        plt.xlim(50, 100)  # Set the x-axis range from 50 to 0
        plt.ylim(49, 100)  # Set the y-axis range from 100 to 50
        plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.legend(loc="lower left", fontsize=32)
        plt.savefig("Prec-recall.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.show()
    else:
        print("The CSV file does not contain the required 'Precision' and 'Recall' columns.")


# Specify the threshold range
threshold_min = 0  # Replace with your desired minimum threshold
threshold_max = 100  # Replace with your desired maximum threshold

# Call the function to plot
plot_precision_recall_by_threshold(data, threshold_min, threshold_max)
