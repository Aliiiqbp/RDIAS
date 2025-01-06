import pandas as pd
import matplotlib.pyplot as plt


def plot_roc_curve(tpr_file, fpr_file):
    # Load the TPR and FPR data
    tpr_data = pd.read_csv(tpr_file)
    fpr_data = pd.read_csv(fpr_file)

    # Sort both datasets by "File Number" (threshold values) to ensure alignment
    tpr_data = tpr_data.sort_values(by="File Number").reset_index(drop=True)
    fpr_data = fpr_data.sort_values(by="File Number").reset_index(drop=True)

    # Extract the TPR and FPR rates
    tpr_rates = tpr_data["Row Count"]
    fpr_rates = fpr_data["Row Count"]

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rates, tpr_rates, marker='', label='RDIAS', linestyle='-', linewidth=2)

    # Plot the diagonal for reference
    # plt.plot([0, 10], [90, 100], 'r--', label='Random Classifier')

    # Labels and title
    plt.title('', fontsize=36)
    plt.xlabel('False Positive Rate (FPR)', fontsize=36)
    plt.ylabel('True Positive Rate (TPR)', fontsize=36)
    plt.legend(loc="lower right", fontsize=24)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=1)
    plt.xlim(0, 14)  # Set the x-axis range from 50 to 0
    plt.ylim(85, 101)  # Set the y-axis range from 100 to 50
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.show()


plot_roc_curve("TPR-90.csv", "FPR-90.csv")
