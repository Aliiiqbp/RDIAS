import pandas as pd

# Load the CSV file into a DataFrame
input_file = "Mixed-1024_test_watermarking_results.csv"  # Replace with your CSV file name
df = pd.read_csv(input_file)

# Define a function to calculate the number of unique bytes with errors
def calculate_byte_errors(error_indexes):
    if pd.isna(error_indexes) or not error_indexes.strip():
        # Handle rows with no errors
        return 0
    # Convert the error indexes string to a list of integers
    indexes = list(map(int, error_indexes.split(',')))
    # Calculate unique byte numbers by integer division by 8
    unique_bytes = set(index // 8 for index in indexes)
    return len(unique_bytes)

# Apply the function to the "Error Indexes" column
df['byte_error'] = df['Error Indexes'].apply(calculate_byte_errors)

# Save the updated DataFrame back to a new CSV file
output_file = "Mixed_Final_File.csv"  # Replace with your desired output file name
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}.")
