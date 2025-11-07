"""
Melissa Maillot - s4851573
COMP3710 2025S2 - Report
modules.py - contains all neural network and custom loss function code
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
from torchvision import  models
#import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix
#import random
#import itertools
#from PIL import Image
#import os
#import glob
#from pathlib import Path
#import shutil
#import matplotlib.pyplot as plt

class EmbeddingNet(nn.Module):
    """
    Non-pretrained CNN to generate image embeddings.
    Simple classifier head for classification
    """
    def __init__(self, out_dim):
        super(EmbeddingNet, self).__init__()
        
        # load ResNet50 model
        resnet = models.resnet50()

        # change the feature extractor head
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_out = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # 0.5
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # 0.5
            nn.Linear(256, out_dim)
        )

        # classification head
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x):
        # extract features
        x = self.extractor(x)
        # Flatten the feature map
        x = x.view(x.size(0), -1) 
        # Final embedding output
        x = self.fc_out(x)
        
        return x
    
    def classify(self, x):
        # classifiy
        return self.classifier(x)
    

class TripletLoss(nn.Module):
    """
    Triplet loss function based on the distance between embeddings.
    L(A, P, N) = max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin)
    """
    def __init__(self, margin=1.0):
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