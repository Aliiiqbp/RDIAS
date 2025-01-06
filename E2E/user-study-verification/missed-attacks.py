import os
import pandas as pd


def filter_and_merge_csv(directory, threshold):
    """
    Filters rows with 'Total Hamming Distance' less than the given threshold
    from all CSV files in the specified directory, and combines them into a single CSV file.
    """
    # Ensure the given path is a valid directory
    if not os.path.isdir(directory):
        print(f"The path '{directory}' is not a valid directory.")
        return

    # List to store filtered DataFrames
    filtered_data = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(directory, filename)

            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)

                # Check if the column 'Total Hamming Distance' exists
                if 'Total Hamming Distance' in df.columns:
                    # Filter rows where 'Total Hamming Distance' is less than the threshold
                    filtered_df = df[df['Total Hamming Distance'] <= threshold]

                    # Append the filtered DataFrame to the list
                    if not filtered_df.empty:
                        filtered_data.append(filtered_df)
                        print(f"Filtered data from '{filename}'")
                else:
                    print(f"'Total Hamming Distance' column not found in '{filename}'")
            except Exception as e:
                print(f"Error reading '{filename}': {e}")

    # If no data was filtered, print a message and return
    if not filtered_data:
        print("No data found matching the criteria.")
        return

    # Concatenate all filtered DataFrames into a single DataFrame
    merged_df = pd.concat(filtered_data, ignore_index=True)

    # Save the merged DataFrame to 'Missed-Attacks.csv'
    output_path = os.path.join(directory + "/Missed-Attacks", "Missed-Attacks-" + str(threshold) + ".csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Filtered data saved to '{output_path}'")


# Example usage
if __name__ == "__main__":
    for t in [i for i in range(257)]:
        filter_and_merge_csv("../user-study-verification", t)
