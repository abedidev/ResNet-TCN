# ResNet-TCN

PyTorch Implementation of ResNet-TCN Hybrid Architecture For End-To-End Video Classification/Regression [1]

This repository contains the source code for the paper [Improving state-of-the-art in Detecting Student Engagement with Resnet and TCN Hybrid Network](url) by Ali Abedi, and Shehroz S. Khan.

In this work, we formulate user engagement detection as a spatio-temporal video analysis problem. A 2D ResNet [2] extracts spatial features from consecutive video frames, and a TCN [3] analyzes the temporal changes in video frames. The extracted feature vectors (by ResNet) from the consecutive frames are considered as the input to the consecutive time steps of the TCN. A fully-connected layer after the last time step of the TCN outputs the classification labels or regression scores corresponding to the level of engagement.


## Requirments
* Python3
* PyTorch
* Torchvision
* OpenCV
* scikit-learn
* pandas
* numpy



## Code Usage:

The input is training and validation raw frames extracted from videos and placed in separate folders. The address of the folders (containing video frames) and corresponding labels should be provided in two csv files, train.csv and test.csv.
datasets.py and transforms.py read the video frames based on their address in the csv files, preprocess and normalize them, and convert them to PyTorch dataloaders.
