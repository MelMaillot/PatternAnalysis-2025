"""
Melissa Maillot - s4851573
COMP3710 2025S2 - Report
train.py - main file, trains and evaluates the siamese network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import random
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import split_data, TripletDataset
from modules import EmbeddingNet, TripletLoss
from predict import test_set_evaluation

# --- Hyperparameters ---
DATA_ROOT = './data'  # IMPORTANT: structure of the data root should be data-root> train-metadata.csv | image
IMAGE_SIZE = 256
EMBEDDING_DIM = 128
MARGIN = 1.25
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
TRAIN_DATA_SUBSET_FRACTION = 0.3
TRAIN_BATCH_SIZE = 32 
VAL_TEST_BATCH_SIZE = 256

def train_epoch(model, dataloader, criterion, classification_crit, optimizer, scheduler, device):
    """
    Trains one epoch of the model
    Return epoch training metrics: 
    average embedding loss, average classification loss, classification accuracy, ROC AUC, AP Score
    """
    model.train()

    all_labels = []
    all_predictions = []
    all_probs = []
    emb_running_loss = 0.0
    class_running_loss = 0.0
    total_samples = 0
    
    for i, (img_a, img_p, img_n, label_a) in enumerate(dataloader):

        img_a, img_p, img_n, label_a = img_a.to(device), img_p.to(device), img_n.to(device), label_a.to(device)

        optimizer.zero_grad()
        
        # Get embeddings
        emb_a = model(img_a)
        emb_p = model(img_p)
        emb_n = model(img_n)
        # Calculate loss
        emb_loss = criterion(emb_a, emb_p, emb_n)

        # classify anchors
        out_a = model.classify(emb_a)
        # Calculate classification loss
        class_loss = classification_crit(out_a, label_a)

        # total loss and update weights
        loss = emb_loss + class_loss
        loss.backward()
        optimizer.step()
        
        # loss logging
        total_samples += img_a.size(0)
        emb_running_loss += emb_loss.item() * img_a.size(0)
        class_running_loss += class_loss.item() * img_a.size(0)

        # Predictions and Probabilities
        _, preds = torch.max(out_a, 1)
        probs = torch.softmax(out_a, dim=1)[:, 1] # Probability of class 1 (Melanoma)
        all_labels.extend(label_a.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

        if (i + 1) % 50 == 0:
            print(f'Batch {i+1}/{len(dataloader)}, Embedding training loss: {(emb_loss.item()):.4f}, Classification training loss: {(class_loss.item()):.4f}')

        # scheduler step is specifically for the OneCycleLR scheduler used
        # if using a different scheduler, change when the step happens
        if (i + 1) % 100 == 0:
            scheduler.step()

    # calculate metrics
    emb_epoch_loss = emb_running_loss / total_samples
    class_epoch_loss = class_running_loss / total_samples
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs)
    aps = average_precision_score(all_labels, all_probs)
    return emb_epoch_loss, class_epoch_loss, acc, auc, aps

def evaluate(model, dataloader, criterion, classification_crit, device):
    """
    Evaluates on training epoch on the provided data (usually the validation set)
    Return epoch validation metrics: 
    average embedding loss, average classification loss, classification accuracy, ROC AUC, AP Score
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []
    emb_running_loss = 0.0
    class_running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for _, (img_a, img_p, img_n, label_a) in enumerate(dataloader):

            img_a, img_p, img_n, label_a = img_a.to(device), img_p.to(device), img_n.to(device), label_a.to(device)
            
            # Get embeddings
            emb_a = model(img_a)
            emb_p = model(img_p)
            emb_n = model(img_n)
            # Calculate loss
            emb_loss = criterion(emb_a, emb_p, emb_n)

            # classify anchors
            out_a = model.classify(emb_a)
            # Calculate classification loss
            class_loss = classification_crit(out_a, label_a)
            
            # loss logging
            total_samples += img_a.size(0)
            emb_running_loss += emb_loss.item() * img_a.size(0)
            class_running_loss += class_loss.item() * img_a.size(0)

            # Predictions and Probabilities
            _, preds = torch.max(out_a, 1)
            probs = torch.softmax(out_a, dim=1)[:, 1] # Probability of class 1 (Melanoma)
            all_labels.extend(label_a.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())

    # calculate metrics
    emb_epoch_loss = emb_running_loss / total_samples
    class_epoch_loss = class_running_loss / total_samples
    acc = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs)
    aps = average_precision_score(all_labels, all_probs)
    return emb_epoch_loss, class_epoch_loss, acc, auc, aps

def main():
    """
    Run Training on SiameseNet for classification of ISIC 2020 data.
    Training will be preformed and then evaluation results on the trained model will be produced.
    """   

    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n")
    # Set Seed
    SEED = 48515739
    random.seed(SEED)
    np.random.seed(SEED)

    # fetch dataframes referencing the data and split between train, validation and test
    train_samples, val_samples, test_samples = split_data(DATA_ROOT)
    # take only a subset of the training set
    train_samples = train_samples.sample(frac=TRAIN_DATA_SUBSET_FRACTION).reset_index(drop=True)
    print(f"Number of normal samples in training data subset: {train_samples[train_samples["target"]== 0].shape[0]}")
    print(f"Number of melanoma samples in training data subset: {train_samples[train_samples["target"]== 1].shape[0]}")

    # Setup DataLoaders
    # add additional transformations to the training set
    train_dataset = TripletDataset(DATA_ROOT, train_samples,
                                   transform=transforms.Compose([
                                       transforms.RandomRotation(degrees=10, fill=(255, 255, 255)),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
                                    ]))

    val_dataset = TripletDataset(DATA_ROOT, val_samples, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=VAL_TEST_BATCH_SIZE, shuffle=True, num_workers=0)

    # Setup Model, Loss, Optimizer
    model = EmbeddingNet(out_dim=EMBEDDING_DIM).to(device)
    criterion = TripletLoss(margin=MARGIN).to(device)
    classifier_crit = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=LEARNING_RATE, 
                                              steps_per_epoch=train_samples.shape[0]//TRAIN_BATCH_SIZE//100,
                                              epochs=NUM_EPOCHS, 
                                              anneal_strategy="cos")
    
    # --- Training Loop ---
    # metric logging
    best_val_AP_score = -1.0
    emb_train_loss_log = []
    emb_val_loss_log = []
    clas_train_loss_log = []
    clas_val_loss_log = []
    train_accuracy_log = []
    val_accuracy_log = []
    train_ROC_AUC_log = []
    val_ROC_AUC_log = []
    train_AP_score_log = []
    val_AP_score_log = []

    print("\n--- Starting Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        print(f"\n==== Training Epoch {epoch} ====")
        emb_train_loss, class_train_loss, train_acc, train_auc, train_aps = train_epoch(model, train_loader, criterion, classifier_crit, optimizer, scheduler, device)

        # Print training metrics
        print(f"Epoch {epoch} training finished.")
        print(f"Average Training Embedding Loss: {emb_train_loss:.4f}")
        print(f"Average Training Classification Loss: {class_train_loss:.4f}")
        print(f"Training Classification Accuracy: {train_acc:.4f}")
        print(f"Training ROC AUC: {train_auc:.4f}")
        print(f"Training Average Precision Score: {train_aps:.4f}")

        # Log training metrics
        emb_train_loss_log.append(emb_train_loss)
        clas_train_loss_log.append(class_train_loss)
        train_accuracy_log.append(train_acc)
        train_ROC_AUC_log.append(train_auc)
        train_AP_score_log.append(train_aps)
        
        # Evaluate
        emb_val_loss, class_val_loss, val_acc, val_auc, val_aps = evaluate(model, val_loader, criterion, classifier_crit, device)
        
        print("--- Validation phase ---")
        # Print validation metrics
        print(f"Average Validation Embedding Loss: {emb_val_loss:.4f}")
        print(f"Average Validation Classification Loss: {class_val_loss:.4f}")
        print(f"Validation Classification Accuracy: {val_acc:.4f}")
        print(f"Validation ROC AUC: {val_auc:.4f}")
        print(f"Validation Average Precision Score: {val_aps:.4f}")

        # Log validation metrics
        emb_val_loss_log.append(emb_val_loss)
        clas_val_loss_log.append(class_val_loss)
        val_accuracy_log.append(val_acc)
        val_ROC_AUC_log.append(val_auc)
        val_AP_score_log.append(val_aps)

        # Save best model
        # We choose the best model on the basis of the highest validation precison-recall score
        # That way we hope to limit false negatives
        if val_aps > best_val_AP_score:
            print(f"Previous best average precision score: {best_val_AP_score:.4f}")
            best_val_AP_score = val_aps
            print("Saving best model...")
            torch.save(model.state_dict(), (Path(DATA_ROOT) / 'best_siamese_model.pth'))
            
    print("\n--- Training Finished ---")
    print(f"Best Validation Average Precision Score: {best_val_AP_score:.4f}%")

    # --- Training visualisation ---
    # Plot loss over epochs
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(NUM_EPOCHS), emb_train_loss_log, label='Train Loss', color='#97a6c4')
    plt.plot(range(NUM_EPOCHS), emb_val_loss_log, label='Validation Loss', color='#384860')
    plt.title('Embedding Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(NUM_EPOCHS), clas_train_loss_log, label='Train Loss', color='#97a6c4')
    plt.plot(range(NUM_EPOCHS), clas_val_loss_log, label='Validation Loss', color='#384860')
    plt.title('Classification Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig((Path(DATA_ROOT) / 'loss_logs.png'))
    plt.show()
    plt.close()

    # plot metrics over epochs
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(NUM_EPOCHS), train_accuracy_log, label='Train Accuracy', color='#97a6c4')
    plt.plot(range(NUM_EPOCHS), val_accuracy_log, label='Validation Accuracy', color='#384860')
    plt.title('Classification Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(NUM_EPOCHS), train_ROC_AUC_log, label='Train ROC AUC', color='#97a6c4')
    plt.plot(range(NUM_EPOCHS), val_ROC_AUC_log, label='Validation ROC AUC', color='#384860')
    plt.title('ROC AUC over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(NUM_EPOCHS), train_AP_score_log, label='Train AP Score', color='#97a6c4')
    plt.plot(range(NUM_EPOCHS), val_AP_score_log, label='Validation AP Score', color='#384860')
    plt.title('Average Precision Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AP Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig((Path(DATA_ROOT) / 'metrics_logs.png'))
    plt.show()
    plt.close()

    # --- Model evaluation ---
    # Load best model
    model.load_state_dict(torch.load((Path(DATA_ROOT) / 'best_siamese_model.pth')))

    # get evaluation metrics
    test_set_evaluation(model, test_samples, device, DATA_ROOT)

if __name__ == "__main__":
    main()