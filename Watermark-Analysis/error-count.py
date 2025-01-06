import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('Mixed-1024_test_watermarking_results.csv')

# Define the pairs of 'Transformation' and 'Parameter' to remove
# Example:
# pairs_to_remove = [
#     ('Rotation', 90),
#     ('Scaling', 1.5),
#     ('Translation', (10, 20)),
#     # Add more pairs as needed
# ]

pairs_to_remove = [
('JPEG', 50),
('JPEG', 70),
# ('JPEG', 90),
('WebP', 50),
('WebP', 70),
('WebP', 90),
# ('Resize', 0.5),
('Resize', 0.75),
# ('Resize', 1.5),
# ('GaussianNoise', 0.02),
('GaussianNoise', 0.04),
('GaussianNoise', 0.08),
('Saturation', 0.5),
('Saturation', 1.5),
('Saturation', 2),
('Brightness', 0.5),
('Brightness', 1.5),
('Brightness', 2),
('Contrast', 0.5),
('Contrast', 1.5),
('Contrast', 2),
('Sharpness', 0.5),
('Sharpness', 1.5),
('Sharpness', 2),
# ('GaussianBlur', 1),
('GaussianBlur', 3),
('GaussianBlur', 5),
('MedianBlur', 1),
('MedianBlur', 3),
('MedianBlur', 5),
('AverageFilter', 1),
('AverageFilter', 3),
('AverageFilter', 5)
]

# Remove the rows with specified pairs
for transformation_value, parameter_value in pairs_to_remove:
    df = df[~((df['Transformation'] == transformation_value) & (df['Parameter'] == parameter_value))]

# Proceed with the analysis
# Extract the 'bit_error_count' column
bit_error_count = df['bit_error_count']

# Print descriptive statistics
print("Descriptive Statistics for 'bit_error_count':")
print(bit_error_count.describe())

# Calculate additional percentiles
percentiles = [1, 5, 25, 50, 75, 95, 96, 97, 98, 99]
percentile_values = bit_error_count.quantile([p/100 for p in percentiles])
print("\nPercentiles:")
print(percentile_values)

# Plot the distribution using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(bit_error_count, bins=30, kde=True)
plt.title('Distribution of Bit Error Count')
plt.xlabel('Bit Error Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot a boxplot
plt.figure(figsize=(10, 2))
sns.boxplot(x=bit_error_count)
plt.title('Boxplot of Bit Error Count')
plt.xlabel('Bit Error Count')
plt.tight_layout()
plt.show()
