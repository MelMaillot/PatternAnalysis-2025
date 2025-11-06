# Siamese network for the ISIC 2020 Kaggle Challenge classification

Melissa Maillot - s4851573

## Problem

### Data

The ISIC 2020 Kaggle Challenge is a classification problem where skin lesions need to be classified between melonoma and normal. The dataset contains 33126 images. The dataset has severe class imbalance with only 584 melonoma samples and 32542 normal samples.

### Siamese networks

We implement a siamese network with triplet loss to attempt to solve this issue. 

Siamese networks are a type of metric learning model, that aims to compare the similarity of samples. Siamese networks learns to distinguish samples from different classes by using a twin network with equal weights. Both samples are passed in the network which extracts their features, and the distance between extracted embedding vectors are compared.

In the case of classification, the training of the siamese networks aims at distinguishing the different classes in the embedding dimension, such that a classifier can be trained on the embeddings. The siamese is used as a feature extractor that maximises the embedding distnaces between classes such that the classifier may easily distinguish classes in this high dimensional space.

Two types of loss are usually used for training a siamese network: contrastive lost and triplet loss. In this implementation, we use triplet loss. Triplet loss compares the distance between embedings of sample of the same class and a sample from another class. 

## Implementation

### Model

The implemented neural network has two parts: a feature extractor and a classification head.

The feature extratctor architecture is a ResNet50 model (not pre-trained) from the PyTorch library [[1](#references)], with the last layer modified to change the extractor head and embedding dimension. As stated above, triplet loss was used.

The classificaiton head is just a single layer perceptron with an output dimension of two for the two classes. Cross entropy loss was used as the loss function for the classifier head.

### Metrics

Several metrics will be used to understand the model's performance.

First, classification accuracy will be used to get a general idea of the model's performance. However this metric isn't ideal to understand the full performance of the classifier, as the heavy data imbalance makes in unrealiable if the datasubset being considered is not balanced.

Then, we will consider the area under the reciever operating characteristic curve (ROC-AUC). It helps us understand how well the model finds true positives compared to false positives, which can help us better understand whether classifier detects melanoma and whether it incorrectly classifies benign as malignant. 

Also, accorrding to [[2](#references)][[3](#references)][[4](#references)], ROC-AUC is not always ideal for binary classification, especially in our case with high imbalance in the dataset. One of the main issue is that ROC-AUC dose not allow us to correctly gauge the importance of false negatives. However, false negatives are extremely important in this context: an undetected melanoma can evolve into a life-threatening condition. As such false negatives are much more worrying than false positives. Since ROC-AUC fails to totally capture their importance, we will also be considering the area under the precision-recall curve (AUPRC), also called the average precision (AP) score. The precision-recall curve plots the precision (`tp/(tp+fp)`) against the recall (`tp/(tp+fn)`). The recall thus includes the much needed information on false negatives into the metric and can help us gauge whether the model could miss malignant cases. The AP score is implemented in `scikit-learn` [[5](#references)].

### File structure

#### Data downloading and storage

The data used in this project is the preprocessed ISIC 2020 dataset available [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data). In this dataset, the images have been resized to `256x256`. The metadata files only contains the images labels, image names and patient IDs. 

To run the code in this repository, you need to download the dataset from the above kaggle link to the machine that will run the code. Ideally, the downloaded materials should be placed in their own folder. The data needs to be reorganised to fit the following structure:
```
your-data-folder-name/
├── train-metadata.csv
└── image/
    ├── ISIC_0015719.jpg
    ├── ISIC_0052212.jpg
    └── ...
```
This `your-data-folder-name` folder can be placed anywhere in the machine, so long as the path to the folder is passed to the `DATA_ROOT` hyperparameter. The parameter is currently set such that if the folder is named `data`, it should be placed in this location after cloning the repository:
```
PatternAnalysis-2025/recognition/Siamese_Network_Maillot/
│
├── readme_figures/
│   └── ...
│
├── dataset.py
├── modules.py
├── predict.py
├── train.py
├── README.md
│
└── data/
    ├── train-metadata.csv
    └── image/
        ├── ISIC_0015719.jpg
        ├── ISIC_0052212.jpg
        └── ...
```
#### Code files

`dataset.py` contains all the classes required for data manipulation and data loading. This class handles making a 80/10/10 train/validation/test split of the data. It also oversamples the minority class for the training set, such that the training set is balanced. At runtime, the training data will be augmented with rotations, flips and colour jitters. The validation and testing set are not oversampled nor augmented. 

`modules.py` contains the neural network architectures and the triplet loss function implementation.

`train.py` contains the main training loop and it's helper functions. The training loop will save the best model as well as the metric plots to the data location (for ease of ignoring with git if needed, this does not affect the dataloading).

`predict.py` contains code to evaluate the model on the test split of the dataset. 

### Python and dependencies

This project uses Python VERRRRRRR

Additonally, the following packages are required in the following versions:

## Results

## Improvements

This model is far from optimal. The number of false negatives is still much to high. This is an issue as melanoma can evolve into life-threatening conditions if not treated early. In that sense, this current model is unrealiable for unseen data. Several points of improvement may include:
- Training on a larger subset of the data: most of the data is currently not used in training as the computing power to train on the full dataset was not available (training times were too long)
- Changing the triplet loss for batch hard mining triplet loss. [[6](#references)] suggests that batch hard mining is a more efficient way to train the model, as it only uses hard triplets to calculate the loss. Implimenting this loss was attempted but unsuccessful: the model did not learn, it is unknown where the issue stemmed from and there was not enough time to troubleshoot the issue
- More extensive hyper-parameter tuning and more exploration of different augmentation techiniques
- Experimenting with the model architecture

## References
[1] resnet50. Available at: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
[2] The relationship between Precision-Recall and ROC curves. Available at: https://dl.acm.org/doi/10.1145/1143844.1143874
[3] Imbalanced Data? Stop Using ROC-AUC and Use AUPRC Instead. Available at: https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494/
[4] ROC AUC vs Precision-Recall for Imbalanced Data. Available at: https://machinelearningmastery.com/roc-auc-vs-precision-recall-for-imbalanced-data/
[5] `average_precision_score`. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
[6] In Defense of the Triplet Loss for Person Re-Identification. Available at: https://arxiv.org/pdf/1703.07737