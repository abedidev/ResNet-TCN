# ResNet-TCN

PyTorch Implementation of ResNet-TCN Hybrid Architecture for End-to-End Video Classification/Regression

In this work, video classification/regression is formulated as a spatio-temporal data analysis problem. A [2D ResNet](https://pytorch.org/vision/0.8/models.html#torchvision.models.resnet18) extracts spatial features from consecutive video frames, and a [TCN](https://github.com/locuslab/TCN) analyzes the temporal changes in video frames. The extracted feature vectors (by ResNet) from the consecutive frames are considered as the input to the consecutive time steps of the TCN. Fully-connected layers, after the last time step of the TCN, output the predicted classification labels or regression scores.


## Requirments
* Python3
* PyTorch
* Torchvision
* OpenCV
* scikit-learn
* pandas
* numpy


## Code Description:

The input is training and validation raw frames extracted from videos and placed in separate folders (using `extractFramesOpenCV.py`). The address of the folders (containing video frames) and corresponding labels should be provided in two csv files, train.csv and validation.csv as follows
 
```python
path,label
/home/videos/train/826412/8264120240/,0
/home/videos/train/510034/5100342024/,1
/home/videos/train/500067/5000671065/,2
/home/videos/train/882654/88265401300,3
...
```

`datasets.py` and `transforms.py` read the video frames based on their address in the csv files, preprocess and normalize them, and convert them to [PyTorch dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

The ResNet-TCN Hybrid Architecture is in `ResTCN.py`. In the class `ResTCN`, the function `forward` [resnet18](https://pytorch.org/vision/0.8/models.html#torchvision.models.resnet18) extracts features from consecutive frames of video, and [TCN](https://github.com/locuslab/TCN) analyzes changes in the extracted features, and fully-connected layers output the final prediction.

Training and validation phases are performed in `train.py`. Training the ResNet and TCN is performed jointly using [Adam](https://pytorch.org/docs/stable/optim.html) optimization algorithm.

The code has been tested on the [DAiSEE](https://iith.ac.in/~daisee-dataset/), Dataset for Affective States in E-Environments for engagement level classification in online classrooms.




