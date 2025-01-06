import pandas as pd


def calculate_column_averages(file_path, columns):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate and print the average for each specified column
    for col in columns:
        if col in df.columns:
            avg = df[col].mean()
            print(f"Average of {col}: {avg:.5f}")
        else:
            print(f"Column '{col}' not found in the CSV file.")


# Example usage
file_path = 'Amin_test_watermarking_results.csv'  # Update with your CSV file path
columns = ['PSNR', 'SSIM', 'LPIPS']  # Update with the columns you want to average
calculate_column_averages(file_path, columns)
