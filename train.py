import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
from utils import get_dataloader

torch.manual_seed(0)
num_epochs = 100
batch_size = 4
lr = .001
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


dataloader = get_dataloader(batch_size,
                            'train.csv',
                            os.path.join(os.getcwd(), 'images_train'),
                            'test.csv',
                            os.path.join(os.getcwd(), 'images_test'))
dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
print(dataset_sizes, flush=True)


model = ResTCN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=.1)

criterion = nn.CrossEntropyLoss().to(device)
softmax = nn.Softmax()

for epoch in range(num_epochs):

    for phase in ['train', 'test']:

        running_loss = .0
        y_trues = np.empty([0])
        y_preds = np.empty([0])

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for inputs, labels in tqdm(dataloader[phase], disable=True):
            inputs = inputs.to(device)
            labels = labels.long().squeeze().to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.max(softmax(outputs), 1)[1]
            y_trues = np.append(y_trues, labels.data.cpu().numpy())
            y_preds = np.append(y_preds, preds.cpu())

        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]

        print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
            phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)
        print('\nconfusion matrix\n' + str(confusion_matrix(y_trues, y_preds)))
        print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))

