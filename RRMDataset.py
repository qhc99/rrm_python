import torch.utils.data.dataset
import numpy as np
from config import N


class InMemoryImageShuffleDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, :, :, :], self.Y[idx, :]
        perm = np.random.permutation(N)
        X = X[:, :, perm]
        Y = Y[perm]
        # Y = ((Y.reshape([-1,2]))[perm,:]).reshape([-1])
        return X, Y


class InMemoryImageSortDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)
        self.sorted_indices = set()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, :, :, :], self.Y[idx, :]
        if idx not in self.sorted_indices:
            sorted_arg = np.argsort(X[0, 0, :])
            X = X[:, :, sorted_arg]
            Y = Y[sorted_arg]
            self.sorted_indices.add(idx)
            self.X[idx, :, :, :] = X
            self.Y[idx, :] = Y
        bin_Y = np.zeros([N * (N + 1)], dtype=float)
        bin_Y[(Y + np.arange(0, N * (N + 1), N + 1)).astype(int)] = 4
        return X, bin_Y


class InMemoryLinearShuffleDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, :, :, :], self.Y[idx, :]
        perm = np.random.permutation(N)
        X = X[:, :, perm]
        X = X.transpose([0, 2, 1]).reshape([75])
        Y = Y[perm]
        # Y = ((Y.reshape([-1,2]))[perm,:]).reshape([-1])
        return X, Y


class InMemoryPreprocessedImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :, :], self.Y[idx, :]


class InMemoryProcessedLinearDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, 0, :, :], self.Y[idx, :]
        return np.reshape(X, [-1]), Y


class InMemoryLinearSortDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X_arg, Y_arg):
        self.X = X_arg  #
        self.Y = Y_arg  # T.from_numpy(Y_arg).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, Y = self.X[idx, :, :, :], self.Y[idx, :]
        sorted_idx = np.argsort(X[0, 0, :])
        X = X[:, :, sorted_idx]
        X = X.transpose([0, 2, 1]).reshape([5 * N])
        Y = Y[sorted_idx]
        return X, Y
