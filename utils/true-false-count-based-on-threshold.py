import os
import pandas as pd
import matplotlib.pyplot as plt

# Set your threshold here
threshold = 4

# Directory containing the CSV files
directory = 'hash-sensitivity'

# Initialize a dictionary to store the results for each file
results = {}

# Iterate through all the CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        # Check if the 'hamming_distance' column exists in the CSV
        if 'hamming_distance' in df.columns:
            # Apply the threshold
            df['threshold_check'] = df['hamming_distance'] < threshold

            # Count the number of True and False
            true_count = df['threshold_check'].sum()
            false_count = len(df) - true_count

            # Store the results
            results[filename] = {'True': true_count, 'False': false_count}

            # Plotting the results
            labels = ['True', 'False']
            values = [true_count, false_count]

            fig, ax = plt.subplots()
            bars = ax.bar(labels, values, color=['green', 'red'])

            # Add actual values on top of the bars in a faded color
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05 * max(values),
                        f'{value}', ha='center', va='bottom', color='black', alpha=0.7, fontsize=10)

            # Add the threshold as text in the plot
            ax.text(0.5, 0.9, f'Threshold: {threshold}', ha='center', va='center',
                    transform=ax.transAxes, color='blue', fontsize=12, alpha=0.7)

            ax.set_ylabel('Counts')
            ax.set_title(f'True/False Counts for {filename} (Threshold: {threshold})')

            # Save the plot in the same directory
            plot_filename = f"{os.path.splitext(filename)[0]}_threshold_{threshold}.png"
            plot_filepath = os.path.join(directory, plot_filename)
            plt.savefig(plot_filepath)

            plt.close()  # Close the figure to free up memory

print("Plots saved successfully.")
