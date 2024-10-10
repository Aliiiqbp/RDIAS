import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_sensitivity_and_auc(directory, transformation_name):
    # Define the hash function names and corresponding file suffix
    hash_functions = ['average_hash', 'dhash', 'phash', 'whash']
    colors = ['blue', 'orange', 'green', 'red']
    auc_values = {}

    plt.figure(figsize=(10, 6))

    for hash_func, color in zip(hash_functions, colors):
        file_path = os.path.join(directory, f'{hash_func}_{transformation_name}.csv')
        if os.path.exists(file_path):
            # Load the data
            data = pd.read_csv(file_path)

            # Normalize the Hamming distances by the global maximum
            data['Cumulative Images'] = data['Number of Images'].cumsum()
            data['Cumulative Images'] /= data['Cumulative Images'].max()

            # Compute the area under the curve (AUC) using the trapezoidal rule
            auc = np.trapz(data['Cumulative Images'], data['Hamming Distance'])
            auc_values[hash_func] = auc

            # Plot the cumulative distribution of Hamming distance
            plt.plot(data['Hamming Distance'], data['Cumulative Images'],
                     label=f'{hash_func} (AUC={auc:.4f})', color=color, linestyle='-', alpha=0.7)

    # Find the hash function with the minimum AUC value
    print(auc_values)
    min_auc_hash_func = min(auc_values, key=auc_values.get)

    # Update the legend to make the minimum AUC bold
    legend_labels = []
    for hash_func in hash_functions:
        label = f'{hash_func} (AUC={auc_values[hash_func]:.2f})'
        if hash_func == min_auc_hash_func:
            label = r"$\mathbf{" + label + "}$" # f'{label}'  # Make the label bold
        legend_labels.append(label)

    # Set plot labels and title
    plt.xlabel('Hamming Distance')
    plt.ylabel('Cumulative Distribution')
    plt.title(f'Transformation: {transformation_name.replace("_", " ").title()}'.replace('Param', '--- Parameter:'))

    # Add the custom legend
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels, handletextpad=1.5, loc='lower right')

    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save the plot
    output_file = f'{transformation_name}_cumulative_sensitivity_comparison.png'
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')
    plt.show()


# Directory containing the CSV files
directory = 'sensitivity-output/tmp'  # Update this path to your CSV files' directory
transformation_name = 'Inpainting_param_0.2'  # Change this to match your file naming convention

# Generate the plot and compute AUC
plot_cumulative_sensitivity_and_auc(directory, transformation_name)

#
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
#
#
# def plot_cumulative_sensitivity_and_auc(directory, transformation_name):
#     # Define the hash function names and corresponding file suffix
#     hash_functions = ['average_hash', 'dhash', 'phash', 'whash']
#     colors = ['blue', 'orange', 'green', 'red']
#     auc_values = {}
#
#     plt.figure(figsize=(10, 6))
#
#     for hash_func, color in zip(hash_functions, colors):
#         file_path = os.path.join(directory, f'{hash_func}_{transformation_name}.csv')
#         if os.path.exists(file_path):
#             # Load the data
#             data = pd.read_csv(file_path)
#
#             # Normalize the Hamming distances by the global maximum
#             data['Cumulative Images'] = data['Number of Images'].cumsum()
#             data['Cumulative Images'] /= data['Cumulative Images'].max()
#
#             # Compute the area under the curve (AUC) using the trapezoidal rule
#             auc = np.trapz(data['Cumulative Images'], data['Hamming Distance'])
#             auc_values[hash_func] = auc
#
#             # Plot the cumulative distribution of Hamming distance
#             plt.plot(data['Hamming Distance'], data['Cumulative Images'],
#                      label=f'{hash_func} (AUC={auc:.4f})', color=color, linestyle='-', alpha=0.7)
#
#     # Find the hash function with the minimum AUC value
#     min_auc_hash_func = min(auc_values, key=auc_values.get)
#
#     # Update the legend to make the minimum AUC bold
#     legend_labels = []
#     for hash_func in hash_functions:
#         label = f'{hash_func} (AUC={auc_values[hash_func]:.4f})'
#         if hash_func == min_auc_hash_func:
#             label = r"$\mathbf{" + label + "}$"  # Make the label bold using mathtext
#         legend_labels.append(label)
#
#     # Set plot labels and title
#     plt.xlabel('Hamming Distance')
#     plt.ylabel('Cumulative Distribution')
#     plt.title(f'Cumulative Sensitivity of Hash Functions to {transformation_name.replace("_", " ").title()}')
#
#     # Add the custom legend
#     handles, _ = plt.gca().get_legend_handles_labels()
#     plt.legend(handles, legend_labels, handletextpad=1.5, loc='best')
#
#     # Create the zoomed-in inset axes
#     ax = plt.gca()  # Get the current main plot axis
#     axins = zoomed_inset_axes(ax, zoom=1.5, loc='upper right')  # zoom=4 means 4x zoom
#
#     # Plot the same data in the zoomed-in section
#     for hash_func, color in zip(hash_functions, colors):
#         file_path = os.path.join(directory, f'{hash_func}_{transformation_name}.csv')
#         if os.path.exists(file_path):
#             # Load the data
#             data = pd.read_csv(file_path)
#
#             # Normalize the Hamming distances by the global maximum
#             data['Cumulative Images'] = data['Number of Images'].cumsum()
#             data['Cumulative Images'] /= data['Cumulative Images'].max()
#
#             # Plot in the main plot
#             ax.plot(data['Hamming Distance'], data['Cumulative Images'],
#                     label=f'{hash_func} (AUC={auc_values[hash_func]:.4f})', color=color, linestyle='-', alpha=0.7)
#
#             # Plot in the zoomed-in plot
#             axins.plot(data['Hamming Distance'], data['Cumulative Images'],
#                        color=color, linestyle='-', alpha=0.7)
#
#     # Set limits for the zoomed-in area
#     x1, x2 = 50, 0  # Set the x-axis limits of the zoomed-in section
#     y1, y2 = 0.6, 1  # Set the y-axis limits of the zoomed-in section
#     axins.set_xlim(x1, x2)
#     axins.set_ylim(y1, y2)
#
#     # Hide ticks for the zoomed inset
#     axins.set_xticks([])
#     axins.set_yticks([])
#
#     # Draw a box and lines to show where the zoomed-in section is
#     mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
#
#     plt.grid(True, which="both", ls="--", linewidth=0.5)
#
#     # Save the plot with the zoomed-in section
#     output_file = f'{transformation_name}_cumulative_sensitivity_comparison_zoomed.png'
#     plt.savefig(output_file)
#     print(f'Plot with zoom saved to {output_file}')
#     plt.show()
#
#
# # Directory containing the CSV files
# directory = 'sensitivity-output/tmp'  # Update this path to your CSV files' directory
# transformation_name = 'Copy-Move_param_0.1'  # Change this to match your file naming convention
#
# # Generate the plot and compute AUC
# plot_cumulative_sensitivity_and_auc(directory, transformation_name)
#
