import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_all_section(file_path):
    """Load the 'All' section from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {int(k): v for k, v in data.get("All", {}).items()}

def calculate_sensitive_accuracy_for_directory(directory):
    """Calculate accuracy for Sensitive files in the given directory."""
    sensitive_data = {}

    # Load data from JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('all.json') and filename.startswith("Sensitive"):
            file_path = os.path.join(directory, filename)
            sensitive_data = load_all_section(file_path)

    if not sensitive_data:
        print(f"Sensitive data missing in {directory}.")
        return None

    # Convert keys to integers and prepare data for thresholds
    sensitive_keys = np.array(list(map(int, sensitive_data.keys())))
    sensitive_values = np.array(list(sensitive_data.values()))

    thresholds = np.arange(0, 13)
    total_samples = 8000

    # Calculate accuracy for each threshold
    accuracies = []
    for T in thresholds:
        sensitive_wrong = np.sum(sensitive_values[sensitive_keys <= T])
        false_percentage = sensitive_wrong / total_samples * 100
        accuracy = (100 - false_percentage) / 100
        accuracies.append(accuracy)

    return thresholds, accuracies


def plot_sensitive_accuracy_for_all_directories(root_directory):
    """Plot accuracy for Sensitive files for each subdirectory."""
    plt.figure(figsize=(12, 8))

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

    plot_data = []

    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            result = calculate_sensitive_accuracy_for_directory(subdir_path)
            if result:
                thresholds, accuracies = result
                area = np.trapz(accuracies, thresholds)
                plot_data.append({
                    'thresholds': thresholds,
                    'accuracies': accuracies,
                    'area': area,
                    'label': subdir
                })

    # plot_data.sort(key=lambda x: x['area'])

    for idx, data in enumerate(plot_data):
        thresholds, accuracies, label = data['thresholds'], data['accuracies'], data['label']
        print(label)
        color = color_map.get(label, 'black')
        linestyle = line_style.get(label, '-')
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
    plt.ylabel(r'Sensitivity $\eta$', fontsize=48)

    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(handles, labels, fontsize=36, loc='lower right', ncol=2)

    plt.grid(axis='y', color='gray', linestyle='--', linewidth=1, alpha=1)
    plt.savefig("sensitivity-" + str(root_directory) + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.show()


# Example usage:
for x in ["pHash-8-20"]:  # "100", "256"
    plot_sensitive_accuracy_for_all_directories(x)
