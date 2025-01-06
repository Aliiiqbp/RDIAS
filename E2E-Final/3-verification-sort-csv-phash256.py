import pandas as pd
import os


# Function to sort each CSV file by 'Image Name' and save it
def sort_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by the 'Image Name' column
    df_sorted = df.sort_values(by=['Image Name'])

    # Save the sorted DataFrame back to the same file
    df_sorted.to_csv(file_path, index=False)


Datasets = ['user-study']  # "MetFace" "FFHQ1000"

for data in Datasets:
    directory = data + '-verification'

    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        sort_csv(file_path)
