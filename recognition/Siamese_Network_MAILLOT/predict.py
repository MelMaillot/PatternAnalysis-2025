"""
Melissa Maillot - s4851573
COMP3710 2025S2 - Report
predict.py - produces evaluation metrics and plots for models on the test set
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
     roc_auc_score, accuracy_score, average_precision_score, 
     confusion_matrix, ConfusionMatrixDisplay, 
     RocCurveDisplay, roc_curve, 
     PrecisionRecallDisplay, precision_recall_curve
     )
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import SkinDataset

def test_set_evaluation(model, test_samples, device, data_root):
    """
    Gives the following evaluation metrics for the provided model:
        - classification accuracy
        - ROC AUC
        - average precision score
        - sensitivity
        - specificity
    
    Provides and saves graphical displays of:
        - the confusion matrix
        - the ROC curve and the precision-recall curve
        - t-SNE visualisation of embeddings

    It does so on the full test dataset and on a balanced subset of the dataset.
    """

    # Get a balanced sample of the test set
    test_samples_subset = test_samples[test_samples["target"]== 0].sample(n=test_samples[test_samples["target"]== 1].shape[0])
    test_samples_subset = pd.concat([test_samples_subset, test_samples[test_samples["target"]== 1]], ignore_index=True)

    for i in ["Subset", ""]:

        # get the correct dataset
        if i == "Subset":
            test_dataset = SkinDataset(data_root, test_samples_subset, transform=None)
        else:
            test_dataset = SkinDataset(data_root, test_samples, transform=None)

        # get the data loader
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)

        model.eval()
        with torch.no_grad():
            test_all_labels = []
            test_all_embeds = []
            test_all_predictions = []
            test_all_probs = []
            for i, (images, labels) in enumerate(test_loader):
                    images = images.to(device)

                    # Get embeddings
                    embeddings = model(images)

                    # classify embeddings
                    output = model.classify(embeddings)

                    # Predictions and Probabilities
                    _, preds = torch.max(output, 1)
                    probs = torch.softmax(output, dim=1)[:, 1] # Probability of class 1 (Melanoma)
                    test_all_labels.extend(labels.cpu().numpy())
                    test_all_embeds.extend(embeddings.cpu().numpy())
                    test_all_predictions.extend(preds.cpu().numpy())
                    test_all_probs.extend(probs.cpu().numpy())

            # --- calculate metrics ---
            test_acc = accuracy_score(test_all_labels, test_all_predictions)
            test_auc = roc_auc_score(test_all_labels, test_all_probs)
            test_aps = average_precision_score(test_all_labels, test_all_probs)
            conf_matrix = confusion_matrix(test_all_labels, test_all_predictions)
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # get metrics
            print(f"Test{" "+i if i else ""} Classification Accuracy: {test_acc:.4f}")
            print(f"Test{" "+i if i else ""} ROC AUC: {test_auc:.4f}")
            print(f"Test{" "+i if i else ""} Average Precision Score: {test_aps:.4f}")
            print(f"Test{" "+i if i else ""} Sensitivity: {sensitivity:.4f}")
            print(f"Test{" "+i if i else ""} Specificity: {specificity:.4f}") 

            # --- plotting ---
            # confusion matrix
            cm = confusion_matrix(test_all_labels, test_all_predictions)
            cm_display = ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.tight_layout()
            plt.savefig((Path(data_root) / ('confusion_matrix'+('_'+i if i else '')+'.png')))

            # ROC AUC and precision-recall
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fpr, tpr, _ = roc_curve(test_all_labels, test_all_probs)
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
            ax1.set_title((i+" " if i else "")+"ROC curve")
            prec, recall, _ = precision_recall_curve(test_all_labels, test_all_probs)
            pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax2)
            ax2.set_title((i+" " if i else "")+"Precision-Recall curve")
            plt.tight_layout()
            plt.savefig((Path(data_root) / ('ROCAUC_PRC'+('_'+i if i else '')+'.png')))

            # t-SNE manifold
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(np.array(test_all_embeds))
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(np.array(embeddings_2d)[:, 0], np.array(embeddings_2d)[:, 1], c=np.array(test_all_labels))
            plt.colorbar(scatter)
            plt.title((i+" " if i else "")+'t-SNE visualisation of embeddings')
            plt.tight_layout()
            plt.savefig((Path(data_root) / ('testing_tsne_embeddings'+('_'+i if i else '')+'.png')))