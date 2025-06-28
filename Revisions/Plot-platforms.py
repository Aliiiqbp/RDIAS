import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt

# --- Define groups and colors ---
groups = ['Facebook', 'WhatsApp', 'Telegram']
colors = {'WhatsApp': 'red', 'Telegram': 'blue', 'Facebook': 'green'}

# Containers for per‐group ROC and PR data
roc_data = {}
pr_data = {}

# Lists to collect all labels/scores across groups
all_true_list = []
all_scores_list = []

for g in groups:
    # --- Load data for group g ---
    attacked_df = pd.read_csv(f'{g}-att-downloaded.csv')
    original_df = pd.read_csv(f'{g}-org-downloaded.csv')

    # Assign true labels
    attacked_df['true_label'] = 1
    original_df['true_label'] = 0

    # Combine attacked and original into one dataframe
    df = pd.concat([attacked_df, original_df], ignore_index=True)

    # Extract scores and true labels as numpy arrays
    y_scores = df['Total Hamming Distance'].values
    y_true = df['true_label'].values

    # Append to our “all‐groups” lists
    all_true_list.append(y_true)
    all_scores_list.append(y_scores)

    # --- Compute ROC curve data for this group ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data[g] = (fpr, tpr, roc_auc)

    # --- Compute Precision‐Recall curve data for this group ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_data[g] = (precision_vals, recall_vals)

# ---------------------------------------------------
# After processing each group, concatenate everything
# ---------------------------------------------------
all_true = np.concatenate(all_true_list)
all_scores = np.concatenate(all_scores_list)

# Compute pooled ROC curve
fpr_all, tpr_all, _ = roc_curve(all_true, all_scores)
roc_auc_all = auc(fpr_all, tpr_all)

# Compute pooled Precision‐Recall curve
precision_all, recall_all, _ = precision_recall_curve(all_true, all_scores)

# --- Plot ROC Curve for all groups + pooled average ---
plt.figure(figsize=(8, 6))
for g in groups:
    fpr, tpr, roc_auc = roc_data[g]
    plt.plot(
        100 * fpr,
        100 * tpr,
        label=f"{g}",
        linestyle='-',
        linewidth=3,
        color=colors[g]
    )

# Plot pooled (average) ROC in black
plt.plot(
    100 * fpr_all,
    100 * tpr_all,
    linestyle='-',
    linewidth=3,
    color='black',
    label='Average'
)

# Random classifier reference line
plt.plot([0, 50], [50, 100], linestyle='--', color='red', label='Random Classifier')


plt.xlabel('FPR (%)', fontsize=42)
plt.ylabel('TPR (%)', fontsize=42)
plt.xlim(0, 50)
plt.ylim(50, 100)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(loc='lower right', fontsize=28)
plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("ROC-total-platforms-with-avg.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()

# --- Plot Precision ‐Recall Curve for all groups + pooled average ---
plt.figure(figsize=(8, 6))
for g in groups:
    precision_vals, recall_vals = pr_data[g]
    plt.plot(
        100 * recall_vals,
        100 * precision_vals,
        label=f"{g}",
        linestyle='-',
        linewidth=3,
        color=colors[g]
    )


# Plot pooled (average) PR in black
plt.plot(
    100 * recall_all,
    100 * precision_all,
    linestyle='-',
    linewidth=3,
    color='black',
    label='Average'
)

# Random classifier reference line at Precision = 50%
plt.plot([100, 50], [50, 50], linestyle='--', color='red', label='Random Classifier')

plt.xlabel('Recall (%)', fontsize=42)
plt.ylabel('Precision (%)', fontsize=42)
plt.xlim(50, 100)
plt.ylim(49, 100)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(loc='lower left', fontsize=28)
plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("Prec-recall-platforms-with-avg.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()
