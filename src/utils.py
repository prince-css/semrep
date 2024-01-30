from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getCumArgDistImpact(df, name):
    dist_categories = [(2, '<2'), (3, '<3'), (4, '<4'), (5, '<5'), (6, '<6'), (7, '<7'), (np.inf, '7+')]
    # Initialize lists to store metrics
    f1_scores, precisions, recalls = [], [], []
    sample_sizes = []
    # Calculate metrics for each distance category
    for max_dist, label in dist_categories:
        # Subset data based on DIST_SUM
        subset = df[df['DIST_SUM'] < max_dist]
        sample_sizes.append(len(subset))
        # Calculate metrics
        f1 = f1_score(subset['y_true'], subset['y_pred']) if len(subset) > 0 else 0
        precision = precision_score(subset['y_true'], subset['y_pred']) if len(subset) > 0 else 0
        recall = recall_score(subset['y_true'], subset['y_pred']) if len(subset) > 0 else 0
        # Append to lists
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    # Extract labels for x-axis
    x_labels = [label for _, label in dist_categories]
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x_labels, f1_scores, marker='o', label='F1 Score')
    ax1.plot(x_labels, precisions, marker='x', label='Precision')
    ax1.plot(x_labels, recalls, marker='s', label='Recall')
    ax1.set_xlabel('Argument Distance')
    ax1.set_ylabel('Metric Value')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plotting sample sizes
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sample Size', color='purple')  # we already handled the x-label with ax1
    ax2.bar(x_labels, sample_sizes, color='purple', alpha=0.3, label='Sample Size')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='upper right')

    plt.title('F1 Score, Precision, Recall, and Sample Size vs Max Distance')
    plt.savefig(f"../plots/{name}.png")
    plt.show()


def getCatArgDistImpact(df, name):
    # Define the DIST_SUM ranges
    dist_ranges = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, float('inf'))]
    dist_labels = ["<2", "2-3", "3-4", "4-5", "5-6", "6-7", "7+"]

    # Initialize lists to store metrics
    f1_scores = []
    precisions = []
    recalls = []
    sample_sizes = []
    # Calculate metrics for each DIST_SUM range
    for min_dist, max_dist in dist_ranges:
        mask = (df['DIST_SUM'] >= min_dist) & (df['DIST_SUM'] < max_dist)
        sample_size = df.loc[mask, 'DIST_SUM'].count()
        sample_sizes.append(sample_size)
        y_true_filtered = df.loc[mask, 'y_true']
        y_pred_filtered = df.loc[mask, 'y_pred']
        
        # Only calculate metrics if there are predictions in this range
        if len(y_true_filtered) > 0:
            f1 = f1_score(y_true_filtered, y_pred_filtered)
            precision = precision_score(y_true_filtered, y_pred_filtered)
            recall = recall_score(y_true_filtered, y_pred_filtered)
        else:
            f1, precision, recall = 0, 0, 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Creating a secondary axis for the sample size
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting metrics
    ax1.plot(dist_labels, f1_scores, marker='o', label='F1 Score', color='blue')
    ax1.plot(dist_labels, precisions, marker='o', label='Precision', color='green')
    ax1.plot(dist_labels, recalls, marker='o', label='Recall', color='red')
    ax1.set_xlabel('Argument Distance')
    ax1.set_ylabel('Metrics', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plotting sample sizes
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sample Size', color='purple')  # we already handled the x-label with ax1
    ax2.bar(dist_labels, sample_sizes, color='purple', alpha=0.6, label='Sample Size')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='upper right')

    plt.title('F1 Score, Precision, Recall, and Sample Size vs Max Distance')
    plt.savefig(f"../plots/{name}.png")
    plt.show()



    
def getAUCCurve(df, name):
    labels=df["y_true"]
    preds=df["y_pred_1"]
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    # Plotting ROC curve
    plt.figure(figsize=(8, 6),dpi=50)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()  # ensures the plot fits within the figure boundaries
    plt.savefig(f"../plots/{name}.png")


def getPrecisionRecallCurve(df,name):
    precision, recall, _ = precision_recall_curve(df["y_true"], df["y_pred_1"])
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../plots/{name}.png")
    plt.show()
    
def getSubObjHeatmap(df, name):
    dist_categories = [(2, '<2'), (3, '<3'), (4, '<4'), (5, '<5'), (6, '<6'), (7, '<7'), (np.inf, '7+')]
    f1x=[]
    # Calculate metrics for each distance category
    for max_distx, labelx in dist_categories:
        f1y=[]
        # Subset data based on DIST_SUM
        subsetx = df[df['SUBJECT_DIST'] < max_distx]
        # print("***",len(subsetx))
        for max_disty, labely in dist_categories:
            subsety = subsetx[subsetx['OBJECT_DIST'] < max_disty]
            f1 = f1_score(subsety['y_true'], subsety['y_pred']) if len(subsety) > 0 else 0
            f1y.append(f1)
            # print(len(subsety))
        f1x.append(f1y)

    f1x=np.array(f1x)
    x_labels = [label for _, label in dist_categories]
    y_labels = [label for _, label in dist_categories]
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(f1x, )
    plt.title('F1 Score Heatmap by Subject and Object Distance Categories')
    plt.xlabel('Subject Distance')
    plt.ylabel('Object Distance')
    plt.colorbar(label='F1 Score')
    plt.xticks(ticks=range(len(dist_categories)), labels=x_labels)
    plt.yticks(ticks=range(len(dist_categories)), labels=y_labels)
    # Annotating each cell with the F1 score
    for i in range(len(dist_categories)):
        for j in range(len(dist_categories)):
            text = plt.text(i, j, round(f1x[i, j], 4),
                        ha="center", va="center", color="w")
    plt.savefig(f"../plots/{name}.png")
    plt.show()