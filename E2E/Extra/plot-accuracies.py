import pandas as pd
import matplotlib.pyplot as plt

dataset = 'CLIC'  # or 'DIV2K'

# File paths for the 4 CSV files
csv_files = {
    'Setup 1': dataset + '_accuracy_1.csv',
    'Setup 2': dataset + '_accuracy_2.csv',
    'Setup 3': dataset + '_accuracy_3.csv',
    'Setup 4': dataset + '_accuracy_4.csv'
}

# Plot accuracy vs threshold for all 4 experiments
def plot_accuracy_vs_threshold(csv_files):
    plt.figure(figsize=(10, 6))

    # Different colors for each experiment
    colors = ['blue', 'green', 'red', 'purple']

    for i, (exp_name, file_path) in enumerate(csv_files.items()):
        # Load the data
        df = pd.read_csv(file_path)

        # Convert accuracies to percentages if they are in decimal form
        df['Accuracy'] = df['Accuracy'] * 100

        # Plot the accuracy vs threshold
        plt.plot(df['Threshold'], df['Accuracy'], color=colors[i], label=exp_name, marker='o', linewidth=3)
        plt.xticks(df['Threshold'], fontsize=18)

    # Set Y-axis limits from 50 to 100
    plt.ylim(50, 100)

    # Set Y-axis ticks to include 50 and 100
    plt.yticks(range(50, 101, 5), fontsize=18)  # Adjust the step as needed

    # Add labels and title
    plt.xlabel('Threshold', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.title('')
    plt.legend(fontsize=22)
    plt.grid(True, linestyle=':')

    # Display the plot
    plt.show()

# Call the function to plot
plot_accuracy_vs_threshold(csv_files)
