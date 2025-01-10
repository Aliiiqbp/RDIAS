import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width


def load_all_section(file_path):
    """Load the 'All' section from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {int(k): v for k, v in data.get("All", {}).items()}

def calculate_accuracy_for_directory(directory):
    """Calculate overall accuracy for Robust and Sensitive files in the given directory."""
    robust_data = {}
    sensitive_data = {}

    # Load data from JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('all.json'):
            file_path = os.path.join(directory, filename)
            if filename.startswith("Robust"):
                robust_data = load_all_section(file_path)
            elif filename.startswith("Sensitive"):
                sensitive_data = load_all_section(file_path)

    if not robust_data or not sensitive_data:
        print(f"Missing data in {directory}.")
        return None

    # Convert keys to integers and prepare data for thresholds
    robust_keys = np.array(list(map(int, robust_data.keys())))
    robust_values = np.array(list(robust_data.values()))
    sensitive_keys = np.array(list(map(int, sensitive_data.keys())))
    sensitive_values = np.array(list(sensitive_data.values()))

    thresholds = np.arange(0, 13)
    total_samples = 16000

    # Calculate overall accuracy for each threshold
    accuracies = []
    for T in thresholds:
        robust_wrong = np.sum(robust_values[robust_keys > T])
        sensitive_wrong = np.sum(sensitive_values[sensitive_keys <= T])
        false_percentage = (robust_wrong + sensitive_wrong) / total_samples * 100
        accuracy = (100 - false_percentage) / 100
        accuracies.append(accuracy)

    return thresholds, accuracies

def plot_accuracy_for_all_directories(root_directory):
    """Plot overall accuracy for each subdirectory in the root directory."""
    plt.figure(figsize=(12, 8))
    # colors = plt.cm.get_cmap('tab20', 10)  # Generate a color map with 10 colors

    # color_map = {
    #     'aHash': 'black',
    #     'dHash': '#ff700e',
    #     'wHash': 'purple',
    #     'PDQ': 'red',
    #     'pHash': 'blue',
    #     'NeuralHash': 'green',
    # }
    #
    # line_style = {
    #     'aHash': ':',
    #     'dHash': '--',
    #     'wHash': '--',
    #     'PDQ': ':',
    #     'pHash': '--',
    #     'NeuralHash': '--',
    # }

    color_map = {
        '8': 'black',
        '10': '#ff700e',
        '12': 'purple',
        '14': 'red',
        '16': 'blue',
        '18': 'green',
        '20': 'orange'
    }

    line_style = {
        '8': '--',
        '10': '--',
        '12': '--',
        '14': '--',
        '16': '--',
        '18': '--',
        '20': '--'
    }


    # List to store data for plotting
    plot_data = []

    # Iterate over subdirectories and calculate accuracy
    for idx, subdir in enumerate(os.listdir(root_directory)):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            result = calculate_accuracy_for_directory(subdir_path)
            if result:
                thresholds, accuracies = result
                # Calculate area under the curve
                area = np.trapz(accuracies, thresholds)
                # Store data for plotting
                plot_data.append({
                    'thresholds': thresholds,
                    'accuracies': accuracies,
                    'area': area,
                    'label': subdir
                })

    # Sort the plot_data based on area under the curve in ascending order
    # plot_data.sort(key=lambda x: x['area'])

    # Now plot the data in sorted order
    for idx, data in enumerate(plot_data):
        thresholds = data['thresholds']
        accuracies = data['accuracies']
        label = data['label']
        # Use colors in order
        color = color_map.get(label, 'red')
        linestyle = line_style.get(label, '-')
        # For the last one (maximum area), make the line bold
        if int(label) % 2 == 0:
            plt.plot(thresholds, accuracies, marker='', markersize=6, linestyle=linestyle, color=color, label=label,
                 linewidth=5)

    plt.rcParams['font.family'] = 'sans-serif'  # or 'sans-serif', 'monospace', etc.
    plt.rcParams['font.style'] = 'normal'  # or 'normal', 'oblique'
    plt.rcParams['font.weight'] = 'normal'  # or 'normal', 'light'

    plt.xticks(np.arange(0, 13, 2), fontsize=48)
    plt.yticks(fontsize=48)
    plt.ylim(0.4, 1.04)
    plt.xlabel(r'Threshold $\tau$', fontsize=48)
    plt.ylabel('Accuracy', fontsize=48)
    # plt.title('Overall', fontsize=30, fontweight='bold')

    # Get the handles and labels from the current axes
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create the legend, increasing font size and placing it on the lower left
    legend = plt.legend(handles, labels, fontsize=34, loc='lower right', ncol=2)

    plt.grid(axis='y', color='gray', linestyle='--', linewidth=1, alpha=1)
    plt.savefig("overall-" + str(root_directory) + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.show()

# Example usage:
# Specify the root directory containing multiple subdirectories
for x in ["pHash-8-20"]:  # "100", "256"
    plot_accuracy_for_all_directories(x)
