import pandas as pd
import matplotlib.pyplot as plt

legend_properties = {'weight':'bold', 'size': 30}


# Function to plot TP and FP rates for a given dataset
def plot_rates(file_path, title):
    # Load the data
    df = pd.read_csv(file_path)

    # Multiply by 100 to scale to percent
    df['TP'] *= 100
    df['FP'] *= 100
    df['FP'] = 100 - df['FP']

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot TP Rate with line and markers
    plt.plot(
        df['Threshold'],
        df['TP'],
        color='green',
        label='TP Rate',
        marker='o',
        linestyle='-',
        linewidth=4
    )

    # Plot FP Rate with line and markers
    plt.plot(
        df['Threshold'],
        df['FP'],
        color='red',
        label='TN Rate',
        marker='o',
        linestyle='-',
        linewidth=4
    )

    # Removed the annotations (numbers) from the plot
    plt.xticks(df['Threshold'], fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    plt.xlabel('Threshold', fontsize=22, fontweight='bold')
    plt.ylabel('Rate', fontsize=22, fontweight='bold')
    plt.ylim(-5, 105)
    plt.title(title)
    plt.legend(loc='lower center', prop=legend_properties)
    plt.grid(True, linestyle=':')
    plt.show()

# Plot for the two CSV files
div2k_file = 'DIV2K_dataset_6.csv'
clic_file = 'CLIC_dataset_6.csv'

plot_rates(div2k_file, '')
plot_rates(clic_file, '')
