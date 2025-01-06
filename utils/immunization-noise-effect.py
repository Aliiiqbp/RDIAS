import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file
csv_file_path = 'div2k-801-900-jpeg-immune-images-whash-Q/immunization-results-div2k-801-900.csv'

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Count the number of True and False values in the 'hash_match' column
true_count = df['hash_match'].sum()
false_count = len(df) - true_count

# Prepare the data for plotting
labels = ['True', 'False']
counts = [true_count, false_count]
colors = ['green', 'red']

# Create the histogram plot
plt.bar(labels, counts, color=colors)
plt.xlabel('Hash Match')
plt.ylabel('Count')
plt.title('Histogram of Hash Match Values')

# Save the plot in the same directory as the CSV file
output_dir = os.path.dirname(csv_file_path)
output_path = os.path.join(output_dir, 'hash_match_histogram.png')
plt.savefig(output_path)

# Show the plot
plt.show()
