import pandas as pd

# Load the updated CSV file into a DataFrame
input_file = "Mixed_Final_File.csv"  # Replace with your updated CSV file name
df = pd.read_csv(input_file)

# Define a function to calculate the byte index
def calculate_byte_index(byte_error, bit_error_count):
    if byte_error == 0 or bit_error_count == 0:
        # Ensure zero for rows with zero in byte_error or bit_error_count
        return 0
    return byte_error / bit_error_count

# Assume there's a column named "bit_error_count" in the DataFrame
# Apply the function to calculate the byte_index for each row
df['byte_index'] = df.apply(
    lambda row: calculate_byte_index(row['byte_error'], row['bit_error_count']),
    axis=1
)

# Save the updated DataFrame back to a new CSV file
output_file = "byte_index_file.csv"  # Replace with your desired output file name
df.to_csv(output_file, index=False)

print(f"File with byte_index saved as {output_file}.")
