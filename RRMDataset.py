import torch.utils.data.dataset
import numpy as np
from config import N, LESS_N, DUMMY_DATA


class ImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :, :], self.Y[idx, :]


class PaddedImageTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        dummy = np.reshape(np.tile(np.array(DUMMY_DATA), (LESS_N, 1)).T, [1, LESS_N, LESS_N])
        return np.concatenate([self.X[idx, :, :, :], dummy], axis=2), self.Y[idx, :]


class LinearDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, 0, :, :], self.Y[idx, :]
        return np.reshape(X, [-1]), Y
