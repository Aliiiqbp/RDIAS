import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Function to analyze each CSV file and save plots
def analyze_csv(file_path, file_name, threshold):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert Verification Result to categorical with proper ordering
    df['Verification Result'] = pd.Categorical(df['Total Hamming Distance'] <= threshold, categories=[False, True], ordered=True)

    # Plot 1: Histogram of Verification Result (True/False)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Verification Result', hue='Verification Result', multiple='stack',
                 palette={True: 'green', False: 'red'}, shrink=0.8)
    plt.title(f'Histogram of Verification Result for {file_name}')
    plt.xlabel('Verification Result')
    plt.ylabel('Count')
    plt.ylim(0, 562)
    plt.xticks([0, 1], ['False', 'True'])
    plt.savefig(os.path.join(directory + "/plots", f'{file_name}_verification_result_histogram.png'))
    plt.close()

    # Plot 2: Histogram of Hamming Distance
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Hamming Distance'], bins=30)
    plt.title(f'Histogram of Hamming Distance for {file_name}')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Count')
    plt.savefig(os.path.join(directory + "/plots", f'{file_name}_hamming_distance_histogram.png'))
    plt.close()


Datasets = ["CLIC", "DIV2K"]  # "MetFace",
threshold = 8

for data in Datasets:
    directory = data + '-verification'

    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        file_name = os.path.splitext(csv_file)[0]
        analyze_csv(file_path, file_name, threshold)
