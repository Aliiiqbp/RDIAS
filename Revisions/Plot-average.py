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

# --- File paths for attacked and original CSVs ---
# Replace these with your actual file paths
attacked_files = [
    'Facebook-att-downloaded.csv',
    'Telegram-att-downloaded.csv',
    'WhatsApp-att-downloaded.csv'
]
original_files = [
    'Facebook-org-downloaded.csv',
    'Telegram-org-downloaded.csv',
    'WhatsApp-org-downloaded.csv'
]

# --- Load and merge attacked files ---
attacked_dfs = []
for fp in attacked_files:
    df_att = pd.read_csv(fp)
    attacked_dfs.append(df_att)
attacked_df = pd.concat(attacked_dfs, ignore_index=True)
attacked_df['true_label'] = 1

# --- Load and merge original files ---
original_dfs = []
for fp in original_files:
    df_org = pd.read_csv(fp)
    original_dfs.append(df_org)
original_df = pd.concat(original_dfs, ignore_index=True)
original_df['true_label'] = 0

# --- Combine both datasets ---
df = pd.concat([attacked_df, original_df], ignore_index=True)

# Extract scores and labels
y_scores = df['Total Hamming Distance'].values
y_true = df['true_label'].values

# --- Compute metrics for thresholds 0 through 10 ---
metrics = []
for thresh in range(0, 11):
    y_pred = (y_scores > thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics.append({
        'threshold': thresh,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1_score': f1
    })

# Print the results
for m in metrics:
    print(
        f"Threshold = {m['threshold']}: "
        f"Accuracy = {m['accuracy']:.4f}, "
        f"Recall = {m['recall']:.4f}, "
        f"Precision = {m['precision']:.4f}, "
        f"F1 Score = {m['f1_score']:.4f}"
    )

# --- Plot ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(100 * fpr, 100 * tpr, marker='', label='RDIAS', linestyle='-', linewidth=4, color='blue')
plt.plot([0, 50], [50, 100], linestyle='--', color='red', label='Random Classifier')
plt.xlabel('FPR (%)', fontsize=42)
plt.ylabel('TPR (%)', fontsize=42)
plt.xlim(0, 50)  # Set the x-axis range from 50 to 0
plt.ylim(50, 100)  # Set the y-axis range from 100 to 50
plt.title('')
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(loc='lower right', fontsize=32)
plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("ROC-total-all.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()

# --- Plot Precision-Recall Curve ---
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(100 * recall_vals, 100 * precision_vals, marker='', label='RDIAS', linestyle='-', linewidth=4, color='blue')
plt.plot([100, 50], [50, 50], linestyle='--', color='red', label='Random Classifier')
plt.xlabel('Recall (%)', fontsize=42)
plt.ylabel('Precision (%)', fontsize=42)
plt.xlim(50, 100)  # Set the x-axis range from 50 to 0
plt.ylim(49, 100)  # Set the y-axis range from 100 to 50
plt.title('')
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(loc='lower left', fontsize=32)
plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("Prec-recall-all.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()
