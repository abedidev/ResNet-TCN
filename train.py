import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import datasets
import transforms
import torchvision
import numpy as np
import cv2

import transforms_train
from ResTCN import ResTCN


torch.manual_seed(0)
num_epochs = 100
batch_size = 10
lr = .001
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


def writeUnNormalize(data, phase):
    for i in range(data.shape[1]):
        image = data[0, i]
        inv_normalize = torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
        image = inv_normalize(image)
        image = image.cpu().detach().numpy()
        image = np.moveaxis(image, 0, 2)
        image *= 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(images, 'images_' + phase, 'image_' + str(i) + '.png'), image)


def get_dataloader(batch_size, csv, root):
    dataset = datasets.VideoDataset(csv, root, transform=torchvision.transforms.Compose([
        transforms.VideoFolderPathToTensor()]))

    # dataset = torch.utils.data.Subset(dataset, list(range(10)))

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)


def get_dataloader_concat(batch_size, csv, csv_, root):
    dataset = datasets.VideoDataset(csv, root, transform=torchvision.transforms.Compose([
        transforms_train.VideoFolderPathToTensor()]))

    # dataset = torch.utils.data.Subset(dataset, list(range(10)))

    dataset_ = datasets.VideoDataset(csv_, root, transform=torchvision.transforms.Compose([
        transforms.VideoFolderPathToTensor()]))

    # dataset_ = torch.utils.data.Subset(dataset_, list(range(10)))

    dataset = torch.utils.data.ConcatDataset([dataset, dataset_])
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)


model = ResTCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=50, gamma=.1)

criterion = nn.BCEWithLogitsLoss().to(device)

dataloader = {
    'train': get_dataloader_concat(batch_size,
                                   os.path.join(data_path, 'train.csv'),
                                   os.path.join(data_path, 'validation.csv'),
                                   root),
    'val': get_dataloader(batch_size,
                          os.path.join(data_path, 'test.csv'),
                          root)}

dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'val']}
print(dataset_sizes, flush=True)

epoch = 0
phase = 'train'
for phase in ['train', 'val']:
    for inputs, labels in tqdm(dataloader[phase]):
        writeUnNormalize(inputs, phase)
        break

for epoch in range(num_epochs):

    for phase in ['train', 'val']:

        running_loss = .0
        y_trues = np.empty([0])
        y_preds = np.empty([0])

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for inputs, labels in tqdm(dataloader[phase]):
            inputs = inputs.to(device)
            labels = labels.float().squeeze().to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            y_trues = np.append(y_trues, labels.data.cpu().numpy())
            y_preds = np.append(y_preds, preds.detach().cpu().numpy())

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]

        print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
            phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)
        print('\nconfusion matrix\t' + str(confusion_matrix(y_trues, y_preds)))
        print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))
        try:
            print('\nroc auc\t' + str(roc_auc_score(y_trues, y_preds)))
        except:
            print('\nroc auc\t' + '...')

        if phase == 'train' and epoch % 10 == 0:
            torch.save(model, os.path.join(save_path, 'model' + str(epoch) + '.pth'))
