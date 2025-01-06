import pandas as pd
import matplotlib.pyplot as plt

# Read the accuracy CSV files
clic_acc = pd.read_csv('CLIC_accuracy.csv')
div2k_acc = pd.read_csv('DIV2K_accuracy.csv')

# Plot accuracy for DIV2K dataset
plt.figure(figsize=(10, 6))

# Plot accuracy vs Threshold for DIV2K
plt.plot(
    div2k_acc['Threshold'],
    div2k_acc['Accuracy'],
    label='RSIA DIV2K',
    marker='o',
    linestyle='-',
    color='blue',
    linewidth=3
)


plt.plot(
    clic_acc['Threshold'],
    clic_acc['Accuracy'],
    label='RSIA CLIC',
    marker='o',
    linestyle='-',
    color='purple',
    linewidth=3
)

# Add a constant 50% accuracy line for C2PA
plt.hlines(
    y=0.5,
    xmin=min(clic_acc['Threshold'].min(), div2k_acc['Threshold'].min()),
    xmax=max(clic_acc['Threshold'].max(), div2k_acc['Threshold'].max()),
    colors='gray',
    linestyles='--',
    label='C2PA Accuracy',
    linewidth=2
)


# Labeling the axes
plt.xlabel('Threshold', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Adding a title
plt.title('', fontsize=22)

# Adding a legend
plt.legend(fontsize=22, loc='center right')

# Display the plot
plt.show()
