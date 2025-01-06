import pandas as pd
import matplotlib.pyplot as plt

# Load the frequency data from the CSV file
frequency_data = pd.read_csv('error_frequency_table.csv', index_col=0)

# Plot the distribution
plt.figure(figsize=(12, 6))
frequency_data['Frequency'].plot(kind='bar', color='skyblue')
plt.title('Distribution of Errors Across Indexes')
plt.xlabel('Index')
plt.ylabel('Number of Errors')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()
