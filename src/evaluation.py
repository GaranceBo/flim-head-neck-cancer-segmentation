import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import binary_dilation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from .losses import dice_coefficient_continuous

# Evaluation function: accuracy, dice, ROC, confusion matrix, example masks
def evaluate_model(labels, probs, model_name, threshold=0.5, verb=True, dilated_coef=None):
    def evaluate_single(labels, title_suffix=""):
        preds = (probs > threshold).float()
        # Dilated ground truth if requested
        if dilated_coef is not None:
            print("\n==============================")
            print(f"Mask dilation coef {dilated_coef} pixels.")
            print("==============================")
            labels_np = labels.cpu().numpy()
            dilated_np = np.zeros_like(labels_np)
            for i in range(labels_np.shape[0]):
                dilated_np[i] = binary_dilation(labels_np[i].squeeze(), iterations=dilated_coef)
            dilated_labels = torch.tensor(dilated_np, device=labels.device, dtype=labels.dtype)
        else:
            dilated_labels = labels
        # Flatten
        y_true_flat = labels.cpu().numpy().flatten()
        y_pred_flat = preds.cpu().numpy().flatten()
        y_prob_flat = probs.cpu().numpy().flatten()
        # Accuracy / Dice
        acc = accuracy_score(y_true_flat, y_pred_flat)
        dice = dice_coefficient_continuous(labels, preds).item()
        # Recall / Precision
        labels_pos_dil = (dilated_labels == 1)
        tp = ((preds == 1) & labels_pos_dil).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (dilated_labels == 0)).sum().float()
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        recall = recall.item()
        precision = precision.item()
        print(f"\n{title_suffix}")
        print(f"Accuracy for {model_name}, threshold {threshold}: {acc:.4f}")
        print(f"Dice Coefficient for {model_name}, threshold {threshold}: {dice:.4f}")
        print(f"Recall for {model_name}, threshold {threshold}: {recall:.4f}")
        print(f"Precision for {model_name}, threshold {threshold}: {precision:.4f}")
        # ROC + AUC
        try: 
            fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
            auc_score = roc_auc_score(y_true_flat, y_prob_flat)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
            # PR curve + AUPRC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_flat, y_prob_flat)
            auprc = average_precision_score(y_true_flat, y_prob_flat)
            print(f"AUPRC for {model_name}: {auprc:.4f}")
            plt.figure(figsize=(6,6))
            plt.plot(recall_curve, precision_curve, label=f'AUPRC = {auprc:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(e)
        # Probability distributions
        cancer_probs = y_prob_flat[y_true_flat == 1]
        normal_probs = y_prob_flat[y_true_flat == 0]
        plt.figure(figsize=(8,5))
        plt.hist(normal_probs, bins=50, alpha=0.5, density=True, label='Normal GT')
        plt.hist(cancer_probs, bins=50, alpha=0.5, density=True, label='Cancer GT')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title(f"Probability Distributions - {model_name}")
        plt.legend()
        plt.show()
        # Confusion matrix
        try: 
            cm = confusion_matrix(y_true_flat, y_pred_flat)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Cancer', 'Cancer'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for {model_name}')
            plt.show()
        except Exception as e:
            print(e)
        # Example mask
        if verb:
            idx = random.randint(0, labels.size(0)-1)
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            axs[0].imshow(labels[idx].cpu().squeeze(), cmap='gray')
            axs[0].set_title(f'Ground Truth Mask (Index {idx})')
            axs[0].axis('off')
            axs[1].imshow(preds[idx].cpu().squeeze(), cmap='gray')
            axs[1].set_title(f'Predicted Mask (Index {idx})')
            axs[1].axis('off')
            plt.show()

    # One mask or two masks
    if isinstance(labels, tuple):
        print("\n==============================")
        print("Evaluating for unfilled mask")
        print("==============================")
        evaluate_single(labels[0], "Unfilled mask")
        print("\n==============================")
        print("Evaluating for filled mask")
        print("==============================")
        evaluate_single(labels[1], "Filled mask")
    else:
        evaluate_single(labels)
