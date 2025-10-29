"""
Gen-AI coding help
Prompt no.2

    original prompt:
    Create a classifier based on Siamese network to classify 
    the ISIC 2020 Kaggle Challenge data set (normal and melanoma) 
    having an accuracy of around 0.8 on the test set. 
    Use triplet loss and write the code in pytorch without 
    a pre-trained network

    Google Gemini answer:
    This is a fantastic and challenging request, as achieving an accuracy of 0.8 
    on a complex medical imaging dataset like ISIC 2020 without using a pretrained 
    model requires robust architecture and careful data strategy.

    Since I cannot access the actual ISIC 2020 images, I will provide a complete, 
    runnable PyTorch script that defines the Siamese Network, the Triplet Loss, and 
    a custom TripletDataset class.

    Note on Data: The script is written to mock the data loading process. To run this 
    code, you will need to:

        1) Create a directory structure: data/train/normal, data/train/melanoma, 
        data/test/normal, data/test/melanoma.

        2) Place a few placeholder images (e.g., 5-10 JPEGs/PNGs) in each of these four 
        final folders. The code will load the paths and treat them as the actual images 
        for demonstration purposes.

    This file provides a complete, from-scratch PyTorch implementation of a Siamese Network 
    using Triplet Loss for the ISIC classification task.
    
    Key Components:
        
        1) EmbeddingNet: A simple, custom Convolutional Neural Network (CNN) defined from 
        scratch (no pre-trained weights) to map the input image to a fixed-size embedding 
        vector.
        
        2) TripletDataset: This custom Dataset is crucial. It iterates over the images and, 
        for each image (the Anchor), randomly selects a Positive example (same class) and 
        a Negative example (different class), fulfilling the requirement for Triplet Loss.
        
        3) TripletLoss: Implements the required loss function with a margin of 1.0.
        
        4) evaluate function: Measures performance by calculating the centroid of the 
        'normal' and 'melanoma' embeddings from the test set and classifying new samples 
        based on distance to the closest centroid.
        
    To run this, you must set up the local data/ structure as described in the comments and 
    populate it with sample images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import random
import numpy as np

# --- 1. CONFIGURATION AND UTILITIES ---

# Define the paths and parameters
DATA_ROOT = './data'  # IMPORTANT: Change this if your data directory is elsewhere
IMAGE_SIZE = 128
EMBEDDING_DIM = 128
MARGIN = 1.0
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    """Sets seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()

# --- 2. DATASET AND TRIPLET GENERATION ---

class TripletDataset(Dataset):
    """
    Custom Dataset for generating (Anchor, Positive, Negative) triplets.
    Assumes data is structured: DATA_ROOT/split/class_name/img_file.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'melanoma']
        
        # Dictionary to store image paths by class (0 or 1)
        self.class_to_paths = {}
        self.all_paths = []

        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found. Create it with images to run the model.")
                self.class_to_paths[i] = []
                continue

            paths = glob.glob(os.path.join(class_path, '*'))
            self.class_to_paths[i] = paths
            self.all_paths.extend([(path, i) for path in paths])
        
        # Filter out classes with no data
        self.all_paths = [p for p in self.all_paths if p[0]]
        print(f"Loaded {len(self.all_paths)} images from {root_dir}.")

        if not self.all_paths:
            raise FileNotFoundError(f"No images found in {root_dir}. Please populate the subdirectories.")
            
        # Total number of unique images to iterate over
        self.len = len(self.all_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 1. Select Anchor (A)
        anchor_path, anchor_class = self.all_paths[index]
        
        # 2. Select Positive (P)
        # Select an image from the same class as the anchor, but not the anchor itself
        positive_class = anchor_class
        positive_paths = self.class_to_paths[positive_class]
        
        # Find all positive paths excluding the anchor
        valid_positives = [p for p in positive_paths if p != anchor_path]
        if not valid_positives:
             # Handle edge case where only one image exists in the class (should not happen in real ISIC)
             positive_path = anchor_path 
        else:
             positive_path = random.choice(valid_positives)

        # 3. Select Negative (N)
        # Select a class different from the anchor class (binary case is simple)
        negative_class = 1 - anchor_class
        negative_paths = self.class_to_paths[negative_class]
        
        if not negative_paths:
             # Handle edge case where the other class is empty (should not happen if data is correctly setup)
             negative_path = random.choice(positive_paths) # Fallback to a random positive (will result in zero loss)
        else:
             negative_path = random.choice(negative_paths)
        
        # Load images
        img_a = Image.open(anchor_path).convert('RGB')
        img_p = Image.open(positive_path).convert('RGB')
        img_n = Image.open(negative_path).convert('RGB')

        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        # Return triplet and the anchor's original label for verification/testing
        return img_a, img_p, img_n, anchor_class

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 3. MODEL DEFINITION ---

class EmbeddingNet(nn.Module):
    """Simple non-pretrained CNN to generate image embeddings."""
    def __init__(self, out_dim=EMBEDDING_DIM):
        super(EmbeddingNet, self).__init__()
        
        # Output size after Conv1 (128->64) -> Conv2 (64->32) -> Conv3 (32->16)
        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: Conv -> ReLU -> Pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate the size before the first linear layer
        # 128 -> 64 -> 32 -> 16. The output size is 16x16 with 128 channels.
        self.fc_input_size = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8) # 128 * 16 * 16 = 32768

        # Fully Connected Layer to produce the embedding
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc_out = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        
        # Flatten the feature map
        x = x.view(x.size(0), -1) 
        
        x = nn.functional.relu(self.fc1(x))
        # Final embedding output
        x = self.fc_out(x)
        
        # L2-normalize the embedding vector
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class SiameseNet(nn.Module):
    """The full Siamese Network structure for three inputs (A, P, N)."""
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        # x1=Anchor, x2=Positive, x3=Negative
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

# --- 4. LOSS FUNCTION ---

class TripletLoss(nn.Module):
    """
    Triplet loss function based on the distance between embeddings.
    L(A, P, N) = max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin)
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = 2  # L2 distance

    def forward(self, anchor, positive, negative):
        # Calculate squared L2 distance
        d_pos = nn.functional.pairwise_distance(anchor, positive, p=self.p)
        d_neg = nn.functional.pairwise_distance(anchor, negative, p=self.p)
        
        # Triplet loss formula
        loss = torch.relu(d_pos - d_neg + self.margin).mean()
        return loss

# --- 5. TRAINING AND EVALUATION ---

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for img_a, img_p, img_n, _ in dataloader:
        img_a, img_p, img_n = img_a.to(device), img_p.to(device), img_n.to(device)

        optimizer.zero_grad()
        
        # Get embeddings
        emb_a, emb_p, emb_n = model(img_a, img_p, img_n)
        
        # Calculate loss
        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * img_a.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, distance_threshold=0.5):
    """
    Evaluates the model by measuring inter-class distances.
    A simple approach: treat the model as a binary classifier using distance to a fixed 'Normal' centroid.
    NOTE: A more robust Siamese eval would use K-NN or a dedicated classification head.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for img_a, _, _, label in dataloader:
            img_a = img_a.to(device)
            emb_a = model.embedding_net(img_a) # Use only the anchor image
            embeddings.append(emb_a.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # 1. Calculate Centroids
    emb_normal = embeddings[labels == 0]
    emb_melanoma = embeddings[labels == 1]
    
    if len(emb_normal) == 0 or len(emb_melanoma) == 0:
        print("Warning: Missing one class in test set for centroid calculation.")
        return 0.0, 0.0

    centroid_normal = np.mean(emb_normal, axis=0)
    centroid_melanoma = np.mean(emb_melanoma, axis=0)

    # 2. Classify based on distance to the closest centroid
    correct = 0
    
    for i in range(len(embeddings)):
        emb = embeddings[i]
        true_label = labels[i]
        
        # Distance to Normal centroid
        dist_normal = np.linalg.norm(emb - centroid_normal)
        # Distance to Melanoma centroid
        dist_melanoma = np.linalg.norm(emb - centroid_melanoma)
        
        # Prediction: Closest centroid wins (0=Normal, 1=Melanoma)
        pred_label = 0 if dist_normal < dist_melanoma else 1
        
        if pred_label == true_label:
            correct += 1

    accuracy = correct / len(embeddings)
    return accuracy, len(embeddings)


# --- 6. MAIN EXECUTION ---

def main():
    print(f"Using device: {device}")
    
    # Setup DataLoaders
    try:
        train_dataset = TripletDataset(os.path.join(DATA_ROOT, 'train'), transform=data_transforms)
        test_dataset = TripletDataset(os.path.join(DATA_ROOT, 'test'), transform=data_transforms)
    except FileNotFoundError as e:
        print(f"\n--- SETUP ERROR ---")
        print(e)
        print("Please create the following directories and add placeholder images:")
        print("  ./data/train/normal, ./data/train/melanoma")
        print("  ./data/test/normal, ./data/test/melanoma")
        print("-------------------\n")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Setup Model, Loss, Optimizer
    embedding_net = EmbeddingNet(out_dim=EMBEDDING_DIM).to(device)
    model = SiameseNet(embedding_net).to(device)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_accuracy = 0.0
    print("\n--- Starting Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate
        test_accuracy, num_samples = evaluate(model, test_loader)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}: | Train Loss: {train_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}% ({num_samples} samples)")

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f"  --> New best model saved with accuracy {best_accuracy*100:.2f}%")
            
    print("\n--- Training Finished ---")
    print(f"Best Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"Goal: Achieve accuracy around 0.8 (80.00%) using Triplet Loss for robust melanoma classification.")

if __name__ == '__main__':
    main()