# [num. channel, height, width], [num, size]
import numpy as np
from config import N, W
import torch


def seqMappingCost(tasks, schedule):
    time = 0
    cost = 0
    for i in range(0, len(schedule)):
        tk = tasks[:, int(schedule[i])]
        exe_time = max(tk[0], time)
        time = (exe_time + tk[2]) * (exe_time <= (tk[0] + tk[1])) + time * (exe_time > (tk[0] + tk[1]))
        cost = cost + (exe_time - tk[0]) * tk[3] * (exe_time <= (tk[0] + tk[1]))
        cost = cost + tk[4] * (exe_time > (tk[0] + tk[1]))
    return cost


# [height, width], [size] ([0 1 ..0, 1 0 ..., ])
def predictCost(tasks, predict):
    # num = tasks.shape[0]
    schedule = np.zeros([N])
    predict = np.reshape(predict, [N, N + 1])
    indices = set(i for i in range(0, N))
    for i in range(0, N):
        srt = np.argsort(predict[:, i])[::-1]
        while srt[0] not in indices:
            srt = srt[1::]
        schedule[i] = srt[0]
        indices.remove(srt[0])

    return seqMappingCost(tasks, schedule)


def capedActivation(cap=W):
    def f(tensor):
        c = torch.ones(tensor.size()) * cap
        c = c.to('cuda')
        return torch.fmin(tensor, c)

    return f


# outputs and labels shape: [1,110]
def cost_stat(LOADER, _model):
    L = len(LOADER)
    out_cost = np.zeros([L])
    optimal_cost = np.zeros([L])
    for _i, _data in enumerate(LOADER, 0):
        if _i % 100 == 0:
            print(_i)
        _inputs, _labels = _data
        BATCH = _inputs.shape[0]
        tasks = _inputs[0, 0, :, :].cpu().detach().numpy()
        _outputs = _model(_inputs.to('cuda'))
        out_cost[_i] = predictCost(tasks, _outputs[0, :].cpu().detach().numpy())
        optimal_cost[_i] = predictCost(tasks, _labels[0, :].cpu().detach().numpy())

    return out_cost, optimal_cost
