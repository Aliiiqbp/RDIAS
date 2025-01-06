import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import defaultdict


def read_csv(file_path):
    """Read the CSV file."""
    return pd.read_csv(file_path)


def filter_keys(count_dict, key_min, key_max):
    """Filter the dictionary based on a specified range of keys."""
    return {int(k): v for k, v in count_dict.items() if key_min <= int(k) <= key_max}


def plot_transformation(df, transformation_name, key_min=0, key_max=99):
    """Plot the error distribution for a specific Transformation."""
    # Filter the DataFrame for the given Transformation
    filtered_df = df[df['Transformation'] == transformation_name]

    if filtered_df.empty:
        print(f"Transformation '{transformation_name}' not found.")
        return

    # Extract and plot the Error Index Count
    count_json = filtered_df['Error Index Count'].iloc[0]
    count_dict = json.loads(count_json)
    filtered_dict = filter_keys(count_dict, key_min, key_max)

    keys = list(map(int, filtered_dict.keys()))
    values = list(filtered_dict.values())

    for i in range(len(values)):
        values[i] /= max(values)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(keys, values, color='blue')
    plt.xlabel("Indexes", fontsize=32)
    plt.ylabel("Error Frequency", fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title(f"Aggregated Error Distribution for Transformations: {', '.join(transformation_name)}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(str(transformation_name) + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()


def plot_aggregated_transformations(df, transformations, key_min=0, key_max=99):
    """Plot the aggregated error distribution for a subset of Transformations."""
    aggregated_counts = defaultdict(int)

    # Aggregate based on the list of Transformations
    if "All" in transformations:
        subset_df = df
    else:
        subset_df = df[df['Transformation'].isin(transformations)]

    # Sum the counts for each Transformation
    for count_json in subset_df['Error Index Count'].dropna():
        count_dict = json.loads(count_json)
        for key, value in count_dict.items():
            if key_min <= int(key) <= key_max:
                aggregated_counts[int(key)] += value

    # Extract keys and values for plotting
    keys = list(map(int, aggregated_counts.keys()))
    values = list(aggregated_counts.values())

    for i in range(len(values)):
        values[i] /= max(values)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(keys, values, color='blue')
    plt.xlabel("Indexes", fontsize=32)
    plt.ylabel("Error Frequency", fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.title(f"Aggregated Error Distribution for Transformations: {', '.join(transformations)}")
    plt.grid(False)
    plt.tight_layout()

    plt.savefig("All.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()


# Main function to demonstrate usage
if __name__ == "__main__":
    # File path
    file_path = 'Error-Indexes.csv'

    # Read the CSV file
    df = read_csv(file_path)

    # Example usage:
    Tname = [
        # "AverageFilter",
        # "Brightness",
        # "Contrast",
        "GaussianBlur",
        'GaussianNoise',
        "JPEG",
        # "MedianBlur",
        "Resize",
        # "Saturation",
        # "Sharpness",
        # "WebP"
    ]
    # for transformation_name in Tname:
    #     plot_transformation(df, transformation_name)

    # 2. Plot aggregated Transformations
    # transformations_list = ["JPEG", "Resize", "GaussianNoise", "GaussianBlur"]  #
    # plot_aggregated_transformations(df, transformations_list)
    #
    # # 3. Plot for all Transformations
    plot_aggregated_transformations(df, ["All"])

