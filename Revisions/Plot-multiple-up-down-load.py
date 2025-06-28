import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt

# --- Define groups and colors ---
groups = ['F', 'F+T', 'F+T+W']
colors = {'F': 'red', 'F+T': 'blue', 'F+T+W': 'green'}

# Containers for ROC and PR data
roc_data = {}
pr_data = {}

for g in groups:
    # --- Load data for group g ---
    attacked_df = pd.read_csv(f'multiple-platform/{g}-att-downloaded.csv')
    original_df = pd.read_csv(f'multiple-platform/{g}-org-downloaded.csv')

    # Assign true labels
    attacked_df['true_label'] = 1
    original_df['true_label'] = 0

    # Combine attacked and original
    df = pd.concat([attacked_df, original_df], ignore_index=True)

    # Extract scores and true labels
    y_scores = df['Total Hamming Distance'].values
    y_true = df['true_label'].values

    # --- Compute metrics for thresholds 0 through 10 ---
    print(f"\n=== Metrics for group '{g}' ===")
    for thresh in range(0, 11):
        y_pred = (y_scores > thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(
            f"Group {g}, Threshold = {thresh}: "
            f"Accuracy = {acc:.4f}, "
            f"Recall = {rec:.4f}, "
            f"Precision = {prec:.4f}, "
            f"F1 Score = {f1:.4f}"
        )

    # --- Compute ROC curve data ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data[g] = (fpr, tpr, roc_auc)

    # --- Compute Precision-Recall curve data ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_data[g] = (precision_vals, recall_vals)

# --- Plot ROC Curve for all groups ---
plt.figure(figsize=(8, 6))
for g in groups:
    fpr, tpr, roc_auc = roc_data[g]
    plt.plot(
        100 * fpr,
        100 * tpr,
        marker='',
        label=f"{g}",
        linestyle='-',
        linewidth=3,
        color=colors[g]
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
plt.savefig("ROC-multiple-platforms.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()

# --- Plot Precision-Recall Curve for all groups ---
plt.figure(figsize=(8, 6))
for g in groups:
    precision_vals, recall_vals = pr_data[g]
    plt.plot(
        100 * recall_vals,
        100 * precision_vals,
        marker='',
        label=f"{g}",
        linestyle='-',
        linewidth=3,
        color=colors[g]
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
plt.savefig("Prec-recall-multiple-platforms.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()
