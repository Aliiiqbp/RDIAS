import pandas as pd

# Function to compute accuracy and save to CSV
def compute_accuracy_and_save(file_path, output_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Compute accuracy using the formula (TP + 1 - FP) / 2
    df['Accuracy'] = (df['TP'] + (1 - df['FP'])) / 2

    # Save the threshold-accuracy pairs to a CSV file
    df[['Threshold', 'Accuracy']].to_csv(output_path, index=False)
    print(f'Accuracy data saved to {output_path}')


# File paths for the input dataset and output accuracy file
div2k_file = 'DIV2K_dataset_2.csv'  # Replace with your actual file path
clic_file = 'CLIC_dataset_2.csv'    # Replace with your actual file path

# Compute and save accuracy for each dataset
compute_accuracy_and_save(div2k_file, 'DIV2K_accuracy_2.csv')
compute_accuracy_and_save(clic_file, 'CLIC_accuracy_2.csv')
