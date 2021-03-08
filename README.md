# ResNet-TCN

PyTorch Implementation of ResNet-TCN Hybrid Architecture For End-To-End Video Classification/Regression

This repository contains the source code for the paper [Improving state-of-the-art in Detecting Student Engagement with Resnet and TCN Hybrid Network](url) by Ali Abedi, and Shehroz S. Khan.

In this work, we formulate user engagement detection as a spatio-temporal video analysis problem. A [2D ResNet](https://pytorch.org/vision/0.8/models.html#torchvision.models.resnet18) extracts spatial features from consecutive video frames, and a [TCN](https://github.com/locuslab/TCN) analyzes the temporal changes in video frames. The extracted feature vectors (by ResNet) from the consecutive frames are considered as the input to the consecutive time steps of the TCN. Fully-connected layers, after the last time step of the TCN, output the predicted classification labels or regression scores.


## Requirments
* Python3
* PyTorch
* Torchvision
* OpenCV
* scikit-learn
* pandas
* numpy



## Code Description:

The input is training and validation raw frames extracted from videos and placed in separate folders. The address of the folders (containing video frames) and corresponding labels should be provided in two csv files, train.csv and validation.csv as follows
 
```python
path,label
/cluster/videos/train/826412/8264120240/,0
/cluster/videos/train/510034/5100342024/,1
/cluster/videos/train/500067/5000671065/,2
/cluster/videos/train/882654/88265401300,3
...
```

`datasets.py` and `transforms.py` read the video frames based on their address in the csv files, preprocess and normalize them, and convert them to [PyTorch dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

The ResNet-TCN Hybrid Architecture is in `ResTCN.py`. In the class `ResTCN` and the function `forward`, [resnet18](https://pytorch.org/vision/0.8/models.html#torchvision.models.resnet18) extract features from consecutive frames of video, and [TCN](https://github.com/locuslab/TCN) anlayzes changes in the extracted features, and fully-connected layers output the final prediction.

Training and validation phases are performed in `train.py`. Training the ResNet and TCN is performred jointly using [Adam](https://pytorch.org/docs/stable/optim.html) optimzation algorithm.


