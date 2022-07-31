import datetime
import pathlib
import os

import numpy as np
import torch
from scipy.io import loadmat
from config import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LR, EPOCH, \
    RUNNING_LOSS_PERIOD, TRAINING, MODEL_NAME, TRAIN_DATASET_PATH, W, L2, PAD_DATASET_PATH
from type_config import DATASET_TYPE, MODEL_TYPE
from RRMDataset import PaddedImageTestDataset
from torch.utils.data import DataLoader
from func import cost_stat

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.__version__)

# load and transform
data = loadmat(TRAIN_DATASET_PATH)
XTrain = data["XTrain"]
XValidation = data["XValidation"]

YTrain = data["YTrain"]
YValidation = data["YValidation"]

# train
model_name = MODEL_NAME


def WeightedMSELoss(_outputs, _labels):
    msel = torch.nn.MSELoss()
    _outputs = _outputs
    _labels = _labels * W
    return msel(_outputs, _labels)


criterion = WeightedMSELoss
model = MODEL_TYPE() if model_name is None else torch.load(model_name)
print(f'using model: {model_name}')
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

train_set = DATASET_TYPE(XTrain, YTrain)
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_set = DATASET_TYPE(XValidation, YValidation)
test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)

train_time = datetime.datetime.now().strftime('%Y_%m_%d,%H.%M.%S')


def dataset_avg_loss(LOADER, _model=model):
    _running_loss = 0.0
    for _i, _data in enumerate(LOADER, 0):
        _inputs, _labels = _data
        _inputs = _inputs.to(device)
        _labels = _labels.to(device)
        _outputs = _model(_inputs)
        _loss = criterion(_outputs, _labels)
        _running_loss += _loss.item()
    return _running_loss / len(LOADER)


if TRAINING:
    MIN_VALID_LOSS = float("Inf")
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        print("\nepoch start")
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % RUNNING_LOSS_PERIOD == RUNNING_LOSS_PERIOD - 1:
                print(f'[{epoch + 1}:{EPOCH}, {i + 1:5d}] loss: {running_loss / RUNNING_LOSS_PERIOD:.4f}, at '
                      f' {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}')
                running_loss = 0.0

        model.eval()

        valid_loss = dataset_avg_loss(test_loader)
        print(f'[{epoch + 1}] validation loss: {valid_loss:.4f}, at '
              f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        torch.save(model, pathlib.Path(f"./checkpoints/{train_time}__{epoch + 1}_{MODEL_TYPE.__name__}.pt"))
        if valid_loss < MIN_VALID_LOSS:
            MIN_VALID_LOSS = valid_loss
            torch.save(model, pathlib.Path(f"./checkpoints/{train_time}__{MODEL_TYPE.__name__}_BEST.pt"))
            print("save best")
        if epoch != 0:
            os.remove(f"./checkpoints/{train_time}__{epoch}_{MODEL_TYPE.__name__}.pt")

    print('Finished Training')
else:
    model.eval()


    def testLoss():
        print(
            f'training loss: {dataset_avg_loss(DataLoader(train_set, batch_size=TEST_BATCH_SIZE, shuffle=False)):.4f},'
            f' at {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}')
        print(f'validation loss: {dataset_avg_loss(test_loader):.4f}, at '
              f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    def computeScheduleCost():
        train_cost_out, train_cost_optimal = cost_stat(DataLoader(train_set, batch_size=1, shuffle=False), model)
        test_cost_out, test_cost_optimal = cost_stat(DataLoader(test_set, batch_size=1, shuffle=False), model)

        np.save("data/train_cost_out.npy", train_cost_out)
        np.save("data/train_cost_optimal.npy", train_cost_optimal)

        np.save("data/test_cost_out.npy", test_cost_out)
        np.save("data/test_cost_optimal.npy", test_cost_optimal)


    def computeScheduleCost2():
        _data = loadmat(PAD_DATASET_PATH)
        X = _data["X"]
        Y = _data["Y"]
        db = PaddedImageTestDataset(X, Y)
        cost_predict, cost_optimal = cost_stat(DataLoader(db, batch_size=1, shuffle=False), model, flexable=True)

        np.save("data/pad_db_cost_predict_rand_half.npy", cost_predict)
        np.save("data/pad_db_cost_optimal_rand_half.npy", cost_optimal)


    computeScheduleCost2()
