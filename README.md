# ResNet-TCN

# PyTorch Implementation of ResNet-TCN Hybrid Architecture For End-To-End Video Classification/Regression [1]

A 2D ResNet [2] extracts spatial features from consecutive video frames, a TCN [3] analyzes the temporal changes in video frames, and a fully-connected layer outputs classification labels or regression scores. The extracted feature vectors (by ResNet) from the consecutive frames are considered as the input to the consecutive time steps of the TCN. The fully-connected layer after the last time step of the TCN outputs the predicted class.
