import os
import pandas as pd

def process_csv_files(directory):
    # List to store data for the output file
    output_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Extract the number at the end of the file name
            file_number = ''.join(filter(str.isdigit, filename.split('.')[0]))

            # Read the CSV file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # Count the number of rows (excluding the header)
            row_count = len(df) / 10

            # Append the data
            output_data.append([int(file_number), row_count])

    # Create a DataFrame for the output
    output_df = pd.DataFrame(output_data, columns=["File Number", "Row Count"])

    output_df = output_df.sort_values(by="File Number").reset_index(drop=True)


    # Save the DataFrame to a CSV file
    output_file_path = os.path.join(directory, "FPR.csv")
    output_df.to_csv(output_file_path, index=False)

    print(f"Output saved as: {output_file_path}")

# Input directory from the user
process_csv_files('Missed-Commons')
