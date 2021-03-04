from torch.utils.data import Dataset
import pandas as pd
import os


class VideoDataset(Dataset):

    def __init__(self, csv_file, root, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

        self.root = root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        label = self.dataframe.iloc[index].label

        video = os.path.join(self.root,
                             self.dataframe.iloc[index].path)
        video = self.transform(video)

        if label == 0 or label == 1:
            label = 0
        else:
            label = 1

        return video, label
