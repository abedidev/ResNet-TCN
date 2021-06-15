import torch
import torchvision
import cv2
import os


class VideoFolderPathToTensor(object):

    def __init__(self, max_len=None):
        self.max_len = max_len

    def __call__(self, path):

        file_names = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        frames_path = [os.path.join(path, f) for f in file_names]

        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Resize([224, 224])
        ])

        # EXTRACT_FREQUENCY = 18
        EXTRACT_FREQUENCY = 1

        # num_time_steps = int(num_frames / EXTRACT_FREQUENCY)

        num_time_steps = 16
        # num_time_steps = 4

        # (3 x T x H x W), https://pytorch.org/docs/stable/torchvision/models.html
        frames = torch.FloatTensor(channels, num_time_steps, 224, 224)

        for index in range(0, num_time_steps):
            frame = cv2.imread(frames_path[index * EXTRACT_FREQUENCY])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)  # (H x W x C) to (C x H x W)
            frame = frame / 255
            if frame.shape[2] != 224:
                frame = frame[:, :, 80:560]
            frame = transform(frame)
            frames[:, index, :, :] = frame.float()

        return frames.permute(1, 0, 2, 3)
