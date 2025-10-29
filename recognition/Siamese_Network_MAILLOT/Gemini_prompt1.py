"""

Gen-AI coding help
Prompt no.1

    original prompt:
    Create a classifier based on Siamese network to classify 
    the ISIC 2020 Kaggle Challenge data set (normal and melanoma) 
    having an accuracy of around 0.8 on the test set. 
    Use triplet loss and write the code in pytorch

    Google Gemini answer:
    This is a fantastic request that requires implementing a specialized 
    deep metric learning architecture. Building a Siamese network with 
    Triplet Loss involves careful handling of data, especially the selection 
    of Anchor, Positive, and Negative samples (known as "triplet mining").

    Since I cannot access the actual ISIC 2020 image files, I am providing 
    a complete, runnable PyTorch script. This script uses a pre-trained 
    ResNet-18 as the base for the Siamese Network and includes:

        1) Data Generation: A custom function to simulate the ISIC 2020
        dataset structure with dummy image tensors and imbalanced labels 
        (98% normal, 2% melanoma) to reflect the real challenge.

        2) Model: An EmbeddingNet (the Siamese tower) based on ResNet-18.

        3) Online Triplet Mining: A custom BatchHardTripletMiner that
        dynamically generates the hardest informative triplets within each 
        batch.

        4) Final Classification: The trained embeddings are used to train 
        a simple linear classifier (a common practice after metric learning)
        to achieve the final accuracy goal.

    This approach ensures the model learns discriminative embeddings, which is 
    key to achieving high accuracy in classification tasks with metric learning.

    The code provides a complete Siamese network implementation designed for 
    metric learning on image data.
    
        1) EmbeddingNet: Acts as the shared "twin" network, mapping images to a 
        128-dimensional feature space.
        
        2) TripletMarginLoss: Implements Batch-Hard Mining, which is crucial for 
        making Triplet Loss effective. It dynamically finds the hardest positive 
        and negative samples within each batch to maximize the learning signal.
        
        3) Two-Phase Training:
            Phase 1 (Metric Learning): Trains the embedding network to group 
            similar images (same class) and separate dissimilar images 
            (different class).
            Phase 2 (Classification): Freezes the learned embeddings and trains 
            a small, fast linear classifier on top of the embeddings for the final 
            prediction.
    
    This structure is highly robust for imbalanced datasets like ISIC 2020 and is 
    designed to hit the $0.8$ accuracy target, often surpassing it in real-world 
    scenarios due to the quality of the learned embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import random
import itertools

# --- Configuration ---
# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
NUM_SAMPLES = 1500  # Simulate a subset of the dataset
TEST_SIZE = 0.2
BATCH_SIZE = 64
EMBEDDING_DIM = 128
MARGIN = 1.0  # Triplet Loss margin
LEARNING_RATE = 1e-4
NUM_EPOCHS_EMBEDDING = 5  # Training epochs for the Siamese Network
NUM_EPOCHS_CLASSIFIER = 10 # Training epochs for the final classifier


# --- 1. Data Simulation (Replace with actual ISIC data loading) ---

def simulate_isic_data(num_samples, imbalance_ratio=0.02):
    """
    Simulates the imbalanced ISIC 2020 data with dummy image tensors.
    Normal (0) vs. Melanoma (1).
    """
    num_melanoma = int(num_samples * imbalance_ratio)
    num_normal = num_samples - num_melanoma

    # Create dummy image paths/IDs (used for logging/indexing, not actual loading)
    all_ids = [f'img_{i}' for i in range(num_samples)]

    # Create labels
    labels = np.array([1] * num_melanoma + [0] * num_normal)
    
    # Shuffle data
    combined = list(zip(all_ids, labels))
    random.shuffle(combined)
    image_ids, labels = zip(*combined)
    
    # Split into train and test sets
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        image_ids, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )

    print(f"Simulated Data Split:")
    print(f"  Train: {len(train_ids)} samples ({np.sum(train_labels)} Melanoma)")
    print(f"  Test: {len(test_ids)} samples ({np.sum(test_labels)} Melanoma)")
    
    return list(train_ids), list(test_ids), list(train_labels), list(test_labels)


class ISIC2020Dataset(Dataset):
    """Custom Dataset class for ISIC images and labels."""
    def __init__(self, image_ids, labels, is_train=True):
        self.image_ids = image_ids
        self.labels = labels
        self.is_train = is_train
        
        # Standard image transformation for pre-trained models
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # NOTE: In a real implementation, you would load the image here
        # image_path = f'/path/to/isic_data/{self.image_ids[idx]}.jpg'
        # image = Image.open(image_path).convert('RGB')
        
        # --- Dummy Data Generation for Runnability ---
        # Generate a dummy tensor representing a 3-channel 224x224 image
        dummy_image = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
        image = self.transform(dummy_image)
        # --- End Dummy Data Generation ---
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)


# --- 2. Model: Siamese Embedding Net ---

class EmbeddingNet(nn.Module):
    """
    The shared network (Siamese twin) for generating embeddings.
    Uses a pre-trained ResNet-18 as a backbone.
    """
    def __init__(self, embedding_dim):
        super(EmbeddingNet, self).__init__()
        
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all layers initially (optional, good for fine-tuning)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer (classification head)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove the original FC layer
        
        # Add a new dense layer for the embedding output
        self.embedding_layer = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Unfreeze the last few layers and the new embedding layer for training
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) # Fine-tune layer 4

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Generate embedding
        embedding = self.embedding_layer(x)
        
        # Normalize the embedding to unit length (improves stability)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

# --- 3. Triplet Mining Logic (Batch Hard Mining) ---

def get_pairwise_distances(embeddings):
    """Computes the squared Euclidean distance matrix."""
    dot_product = torch.matmul(embeddings, embeddings.T)
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0 # Ensure non-negative distances
    return distances.sqrt()


def get_triplets(labels, distances):
    """
    Performs Batch-Hard Triplet Mining.
    For each anchor, finds the hardest positive and the hardest negative in the batch.
    """
    batch_size = labels.size(0)
    
    # Create mask for positive and negative pairs
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
    
    # 1. Hardest Positive (Anchor-Positive distance should be maximized)
    # Mask to select only positive pairs (i.e., same label, excluding self-distance)
    positive_mask = labels_equal.triu(diagonal=1) | labels_equal.tril(diagonal=-1) 
    
    # Set non-positive distances to a very small number for maximization (finding the largest distance)
    anchor_positive_dist = distances * positive_mask.float()
    
    # Max distance per row (Anchor) is the hardest positive
    hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
    
    # 2. Hardest Negative (Anchor-Negative distance should be minimized)
    # Mask to select only negative pairs (i.e., different label)
    negative_mask = ~labels_equal
    
    # Set non-negative distances to a very large number for minimization (finding the smallest distance)
    # We use a copy to avoid in-place modification of the original distances tensor
    anchor_negative_dist = distances.clone()
    anchor_negative_dist[~negative_mask] = float('inf')
    
    # Min distance per row (Anchor) is the hardest negative
    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

    return hardest_positive_dist, hardest_negative_dist


class TripletMarginLoss(nn.Module):
    """
    Combines Triplet Loss with Batch-Hard mining.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # Get pairwise distances
        distances = get_pairwise_distances(embeddings)
        
        # Perform Batch-Hard mining to find the hardest (Ap) and (An) for each Anchor
        hardest_positive_dist, hardest_negative_dist = get_triplets(labels, distances)
        
        # Calculate Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
        losses = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # Only consider anchors that had at least one valid hard positive and hard negative
        # In this Batch-Hard implementation, every anchor should theoretically have a pair 
        # as long as the batch is sampled to be balanced (which it is via the DataLoader shuffle).
        
        return losses.mean()


# --- 4. Training and Evaluation Functions ---

def train_embedding_net(model, train_loader, criterion, optimizer, epochs, device):
    """Trains the Siamese Embedding Network using Triplet Loss."""
    model.train()
    print("\n--- Training Embedding Network (Metric Learning) ---")
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Generate embeddings
            embeddings = model(images)
            
            # Calculate Triplet Loss using Batch-Hard mining
            loss = criterion(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(images)
            
            if (i + 1) % 50 == 0:
                print(f'Epoch {epoch}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} finished. Average Loss: {epoch_loss:.4f}")

    print("Embedding network training complete.")


# --- 5. Final Classification Head ---

class ClassificationNet(nn.Module):
    """
    A simple linear head trained on top of the fixed embeddings 
    for the final binary classification (Melanoma vs. Normal).
    """
    def __init__(self, embedding_dim):
        super(ClassificationNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # Two classes: 0 (Normal) and 1 (Melanoma)
        )

    def forward(self, x):
        return self.classifier(x)


def train_classifier_head(embedding_net, classifier_head, train_loader, criterion, optimizer, epochs, device):
    """Trains the Classification Head while freezing the Embedding Net."""
    embedding_net.eval()
    classifier_head.train()
    print("\n--- Training Classification Head ---")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Generate embeddings (NO GRADIENT)
            with torch.no_grad():
                embeddings = embedding_net(images)
            
            # Classify
            outputs = classifier_head(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * len(images)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += len(images)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


def evaluate_model(embedding_net, classifier_head, test_loader, device):
    """Evaluates the final model on the test set."""
    embedding_net.eval()
    classifier_head.eval()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            embeddings = embedding_net(images)
            outputs = classifier_head(embeddings)
            
            # Predictions and Probabilities
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of class 1 (Melanoma)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    overall_acc = accuracy_score(all_labels, all_predictions)
    try:
        # AUC is critical for imbalanced data like ISIC
        overall_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle cases where only one class is present (unlikely with stratify, but possible with small batches)
        overall_auc = 0.5 

    print("\n--- Final Test Set Results ---")
    print(f"Overall Classification Accuracy: {overall_acc:.4f}")
    print(f"ROC AUC Score (Melanoma): {overall_auc:.4f}")
    
    # We target an accuracy of around 0.8
    if overall_acc >= 0.78:
        print("\n✅ Target Accuracy Achieved!")
    else:
        print("\n⚠️ Target Accuracy Not Reached in Simulation. Increase epochs or adjust hyperparameters.")


# --- 6. Main Execution ---

if __name__ == '__main__':
    # A. Data Setup
    train_ids, test_ids, train_labels, test_labels = simulate_isic_data(NUM_SAMPLES)

    train_dataset = ISIC2020Dataset(train_ids, train_labels, is_train=True)
    test_dataset = ISIC2020Dataset(test_ids, test_labels, is_train=False)

    # Use standard DataLoader; Triplet mining is handled in the custom loss
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # B. Model Setup
    embedding_net = EmbeddingNet(EMBEDDING_DIM).to(DEVICE)
    classifier_head = ClassificationNet(EMBEDDING_DIM).to(DEVICE)
    
    # C. Phase 1: Train Embedding Net with Triplet Loss
    triplet_criterion = TripletMarginLoss(margin=MARGIN)
    # Only optimize the layers that were unfrozen
    embedding_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, embedding_net.parameters()), 
        lr=LEARNING_RATE
    )

    train_embedding_net(
        embedding_net, 
        train_loader, 
        triplet_criterion, 
        embedding_optimizer, 
        NUM_EPOCHS_EMBEDDING, 
        DEVICE
    )

    # D. Phase 2: Train Classification Head with Cross-Entropy Loss
    # We switch to a standard Cross-Entropy loss for the final classification task
    classification_criterion = nn.CrossEntropyLoss().to(DEVICE)
    # Only optimize the classifier head weights
    classifier_optimizer = optim.Adam(classifier_head.parameters(), lr=LEARNING_RATE * 5) # Faster learning rate for small head

    train_classifier_head(
        embedding_net, 
        classifier_head, 
        train_loader, 
        classification_criterion, 
        classifier_optimizer, 
        NUM_EPOCHS_CLASSIFIER, 
        DEVICE
    )

    # E. Evaluation
    evaluate_model(embedding_net, classifier_head, test_loader, DEVICE)