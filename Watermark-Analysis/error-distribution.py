import pandas as pd
import json
import re
from collections import Counter, OrderedDict

# Read the CSV file
df = pd.read_csv('Mixed-1024_test_watermarking_results.csv')

# Function to count and sort error indexes per group
def count_and_sort_error_indexes(series):
    # Concatenate all 'Error Indexes' strings in the series into one string
    all_error_indexes = ' '.join(series.dropna().astype(str))
    # Use regular expression to split on commas or spaces (one or more)
    error_list = re.split(r'[,\s]+', all_error_indexes.strip())
    # Remove empty strings and filter error indexes within the range 0-399
    error_list = [error for error in error_list if error and error.isdigit() and 0 <= int(error) <= 399]
    # Count occurrences of each error index
    counts = Counter(error_list)
    # Sort the counts dictionary based on integer keys
    sorted_counts = OrderedDict(sorted(counts.items(), key=lambda x: int(x[0])))
    # Convert the sorted dictionary to a JSON string
    counts_json = json.dumps(sorted_counts)
    return counts_json

# Group by 'Transformation' and apply the counting and sorting function
df_counts = df.groupby('Transformation')['Error Indexes'].apply(count_and_sort_error_indexes).reset_index()

# Rename the column to 'Error Index Count'
df_counts = df_counts.rename(columns={'Error Indexes': 'Error Index Count'})

# Save the final output to a new CSV file
df_counts.to_csv('Error-Indexes.csv', index=False)

# Optional: Print the resulting DataFrame
print(df_counts)
