import pandas as pd
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

# Read the CSV files
clic = pd.read_csv('CLIC.csv')
div2k = pd.read_csv('DIV2K.csv')

# Compute TNR for CLIC dataset
clic['TNR'] = 1 - clic['FPR']

# Plot for CLIC dataset
plt.figure(figsize=(10, 6))

# Plot TPR and TNR for CLIC dataset with green and red colors
plt.plot(clic['Threshold'], clic['TPR'], label='RSIA TPR', marker='o', linestyle='-', color='green', linewidth=3)
plt.plot(clic['Threshold'], clic['TNR'], label='RSIA TNR', marker='o', linestyle='-', color='red', linewidth=3)

c2pa_tpr = [0] * len(clic['Threshold'])
c2pa_tnr = [1] * len(clic['Threshold'])
# Plot C2PA TPR and TNR
plt.plot(clic['Threshold'], c2pa_tpr, label='C2PA TPR', linestyle='--', color='green', linewidth=2)
plt.plot(clic['Threshold'], c2pa_tnr, label='C2PA TNR', linestyle='--', color='red', linewidth=2)


# Labeling the axes
plt.xlabel('Threshold', fontsize=22)
plt.ylabel('Rate', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a title
plt.title('')

# Adding a legend
plt.legend(fontsize=22)

# Adding grid lines
plt.grid(False)

# Display the plot
plt.show()

# Compute TNR for DIV2K dataset
div2k['TNR'] = 1 - div2k['FPR']

# Plot for DIV2K dataset
plt.figure(figsize=(10, 6))

# Plot TPR and TNR for DIV2K dataset with green and red colors
plt.plot(div2k['Threshold'], div2k['TPR'], label='RSIA TPR', marker='o', linestyle='-', color='green', linewidth=3)
plt.plot(div2k['Threshold'], div2k['TNR'], label='RSIA TNR', marker='o', linestyle='-', color='red', linewidth=3)

# Plot C2PA TPR and TNR
plt.plot(div2k['Threshold'], c2pa_tpr, label='C2PA TPR', linestyle='--', color='green', linewidth=2)
plt.plot(div2k['Threshold'], c2pa_tnr, label='C2PA TNR', linestyle='--', color='red', linewidth=2)


# Labeling the axes
plt.xlabel('Threshold' , fontsize=22)
plt.ylabel('Rate', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Adding a title
plt.title('')

# Adding a legend
plt.legend(fontsize=22)

# Adding grid lines
plt.grid(False)

# Display the plot
plt.show()
