import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the directory where your CSV files are stored
directory = 'hash-sensitivity'

# Loop over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Create a count plot for the 'hamming_distance' column
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x='hamming_distance', palette='viridis')

        # Set x-axis range from 0 to 150
        plt.xlim(0, 30)

        # Set y-axis range from 0 to 100
        plt.ylim(0, 100)

        # Add title and labels
        plt.title(f'Count Plot of Hamming Distance for {filename}')
        plt.xlabel('Hamming Distance')
        plt.ylabel('Frequency')

        # Annotate each bar with the actual frequency count in a faded color
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=10,
                color='gray',
                alpha=0.7
            )

        # Save the plot with the same name as the CSV file, but with a .png extension
        plot_filename = f"{os.path.splitext(filename)[0]}_hamming_distance_countplot.png"
        plt.savefig(os.path.join(directory, plot_filename))

        # Close the plot to avoid memory issues
        plt.close()
