"""
Melissa Maillot - s4851573
COMP3710 2025S2 - Report
dataset.py - contains all data manipulation and Dataset code
"""

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import Dataset#, DataLoader
from torchvision import transforms#, models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix
import random
#import itertools
from PIL import Image
#import os
#import glob
from pathlib import Path
#import shutil
#import matplotlib.pyplot as plt

SEED = 48515739
random.seed(SEED)
np.random.seed(SEED)


def split_data(data_root):
    """
    Fetches reference dataframe
    Splits data frame in 80/10/10 train/validation/test sets
    Oversamples the minority class to have equal number of each class in the train set
    Returns three dataframes: the train set, the validation set, the test set

    Image files are not manipulated as it would cause unnecessary overhead
    """
    data_dir = Path(data_root)

    # Fetch the image names and labels dataset and load to a dataframe
    data_df = pd.read_csv((data_dir / "train-metadata.csv"), index_col=0)

    # Get IDs and labels for dataset train/validation/test splitting
    # The isic_id is unique
    image_ids = data_df["isic_id"]
    labels = data_df["target"]

    # Split into train, validation and test sets
    # 80% of data to train, 10% to validate, 10% to test
    # Split train and validation/test
    train_ids, val_test_ids, train_labels, val_test_labels = train_test_split(
        image_ids, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    # Split validation and test
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        val_test_ids, val_test_labels, test_size=0.5, stratify=val_test_labels, random_state=SEED
    )

    # Subset dataframe for train, validation and test
    # The isic_id column will be used to fetch the images when dataloading
    # The dataframe index is reset for ease of access at dataloading phase
    train_samples = data_df[data_df["isic_id"].isin(train_ids)].reset_index(drop=True)
    val_samples = data_df[data_df["isic_id"].isin(val_ids)].reset_index(drop=True)
    test_samples = data_df[data_df["isic_id"].isin(test_ids)].reset_index(drop=True)

    # Oversample the minority class in the training set
    # There will be an equal amount of rows for each class
    normal_samples_size = train_samples[train_samples["target"]== 0].shape[0]
    melanoma_sample = train_samples[train_samples["target"]== 1]
    oversample_sample = melanoma_sample.sample(n=normal_samples_size - melanoma_sample.shape[0], replace=True, random_state=SEED)

    # Concatenate the data and the oversaampled data into one dataframe
    train_samples = pd.concat([train_samples, oversample_sample], ignore_index=True)
    train_samples = train_samples.sample(frac=1).reset_index(drop=True)
    # Logic: We duplicate some of the image references in the training data 
    # label dataframe. Since the images will be transformed when loaded, this 
    # will augment the melanoma samples. We only add duplicated rows as this 
    # array is what gets iterated on by the Dataloader. There is no need to 
    # duplicate the image, that is useless use of memory. The augmented array 
    # is shuffled so that randomisation is ensured when dataloaders iterate 
    # the dataset.

    return train_samples, val_samples, test_samples


class TripletDataset(Dataset):
    """
    Custom Dataset for generating (Anchor, Positive, Negative) triplets
    when given the pandas dataframe that lists the images in the set and its labels
    """
    def __init__(self, root_dir, items_df, transform=None):
        #self.root_dir = root_dir
        #self.transform = transform
    
        # get the image folder path
        self.image_dir = (Path(root_dir) / 'image')
        # get the labels dataframe
        self.items_df = items_df
        # Label names
        self.classes = ['normal', 'melanoma']

        # Standard image transformation to which we add supplied tranformations
        self.transform = transforms.Compose(
            (transform.transforms if transform else [])+
            #[transforms.ToPILImage(),
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
            
        # Total number of unique images to iterate over
        self.len = self.items_df.shape[0] 

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 1. Select Anchor (A)
        #anchor_path, anchor_class = self.all_paths[index]

        # Get image information from the dataframe
        anchor = self.items_df.iloc[index]
        # Get image label
        anchor_class = anchor["target"]
        # Get image
        anchor_name = anchor["isic_id"]
        anchor_image = Image.open(self.image_dir / (anchor_name + ".jpg")).convert('RGB')
        # Transform image
        anchor_image = self.transform(anchor_image)
        
        # 2. Select Positive (P)
        # Select an image from the same class as the anchor, but not the anchor itself
        try:
            positive = self.items_df[(self.items_df["isic_id"]!=anchor_name) & (self.items_df["target"]==anchor_class)].sample()
        except:
            # Handle edge case where only one image exists in the class (should not happen in real ISIC)
            positive = anchor

        # Get image
        positive_name = positive["isic_id"].item()
        positive_image = Image.open(self.image_dir / (positive_name + ".jpg")).convert('RGB')
        # Transform image
        positive_image = self.transform(positive_image)

        # 3. Select Negative (N)
        # Select a class different from the anchor class (binary case is simple)
        negative_class = 1 - anchor_class
        # Select a negative sample
        negative = self.items_df[self.items_df["target"]==negative_class].sample()
        # Get image
        negative_name = negative["isic_id"].item()
        negative_image = Image.open(self.image_dir / (negative_name + ".jpg")).convert('RGB')
        # Transform image
        negative_image = self.transform(negative_image)

        # Return triplet and the anchor's original label for verification/testing
        return anchor_image, positive_image, negative_image, anchor_class
    

class SkinDataset(Dataset):
    """
    Custom Dataset to load the test set
    Function the same as TripletDataset, but doesn't return triplets, 
    just an image and its label.
    """
    def __init__(self, root_dir, items_df, transform:transforms.Compose=None):

        # get the image folder path
        self.image_dir = (Path(root_dir) / 'image')
        # get the labels dataframe
        self.items_df = items_df
        # Label names
        self.classes = ['normal', 'melanoma']

        # Standard image transformation to which we add supplied tranformations
        self.transform = transforms.Compose(
            (transform.transforms if transform else [])+
            #[transforms.ToPILImage(),
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )

        self.len = self.items_df.shape[0] 

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        # Get image information from the dataframe
        item = self.items_df.iloc[idx]
        # Get image label
        label = item["target"]
        # Get image
        image_name = item["isic_id"]
        image = Image.open(self.image_dir / (image_name + ".jpg")).convert('RGB')
        # Transform image
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)