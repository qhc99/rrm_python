import torch
from config import N
from func import capedActivation


class RRMConvNet(torch.nn.Module):
    def __init__(self):
        super(RRMConvNet, self).__init__()
        self.net = torch.nn.Sequential(
            # 5*15

            torch.nn.Conv2d(256, 256, (3, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            # 3*15

            torch.nn.Conv2d(256, 256, (3, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            # 1*15

            torch.nn.Conv2d(256, 512, (1, 2 * N + 1), padding=(0, N)),
            torch.nn.ReLU(),
            # 1*15

            torch.nn.Flatten(),

            torch.nn.Linear(512 * N, 4096 * 2),

            torch.nn.Linear(4096 * 2, 4096 * 2),
            torch.nn.ReLU(),

            torch.nn.Linear(4096 * 2, (N + 1) * N),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        schedule = self.net(x)
        return schedule


class RRMFastConvNet(torch.nn.Module):
    def __init__(self):
        super(RRMFastConvNet, self).__init__()
        self.sq = torch.nn.Sequential(
            # 5*15

            torch.nn.Conv2d(1, 4, (3, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            # 3*15

            torch.nn.Conv2d(4, 4, (3, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            # 1*15

            torch.nn.Conv2d(4, 4, (1, 2 * N + 1), padding=(0, N)),
            torch.nn.ReLU(),
            # 1*15

            torch.nn.Flatten(),

            torch.nn.Linear(4 * N, 4 * N),
            torch.nn.ReLU(),

            torch.nn.Linear(4 * N, (N + 1) * N),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return capedActivation()(self.sq(x))


class RRMLinearNet(torch.nn.Module):
    def __init__(self):
        super(RRMLinearNet, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(5 * N, 1024*8),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(1024*8, 1024*8),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(1024*8, 1024*8),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(1024*8, 1024*8),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(1024 * 8, 1024 * 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(1024 * 8, N * (N + 1)),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        schedule = self.seq(x)
        return schedule
