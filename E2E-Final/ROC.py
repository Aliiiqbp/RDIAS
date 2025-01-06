import pandas as pd
import matplotlib.pyplot as plt


def plot_roc_curve(tpr_file, fpr_file):
    # Load the TPR and FPR data
    tpr_data = pd.read_csv(tpr_file)
    fpr_data = pd.read_csv(fpr_file)

    # Sort both datasets by "File Number" (threshold values) to ensure alignment
    tpr_data = tpr_data.sort_values(by="Threshold").reset_index(drop=True)
    fpr_data = fpr_data.sort_values(by="Threshold").reset_index(drop=True)

    # Extract the TPR and FPR rates
    tpr_rates = tpr_data["TPR"]
    fpr_rates = fpr_data["FPR"]

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rates, tpr_rates, marker='', label='RDIAS', linestyle='-', linewidth=4, color='blue')

    # Plot the diagonal for reference
    plt.plot([0, 50], [50, 100], 'r--', label='Random Classifier')

    # plt.plot([0, 0, 50], [0, 100, 100], 'g--', label='Optimal Classifier', linewidth=2)


    # Labels and title
    plt.title('', fontsize=48)
    plt.xlabel('FPR (%)', fontsize=42)
    plt.ylabel('TPR (%)', fontsize=42)
    plt.legend(loc="lower right", fontsize=32)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim(0, 50)  # Set the x-axis range from 50 to 0
    plt.ylim(50, 100)  # Set the y-axis range from 100 to 50
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.savefig("ROC-total.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()


plot_roc_curve("TPR.csv", "FPR.csv")
