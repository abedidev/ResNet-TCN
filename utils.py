import torchvision
from torch.utils.data import DataLoader

import datasets
import transforms


def generate_dataloader(batch_size, csv, root):
    dataset = datasets.VideoDataset(csv,
                                    root,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)


def get_dataloader(batch_size, csv_train, root_train, csv_test, root_test):
    return {
        'train': generate_dataloader(batch_size, csv_train, root_train),
        'test': generate_dataloader(batch_size, csv_test, root_test)}
