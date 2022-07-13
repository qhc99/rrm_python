# [num. channel, height, width], [num, size]
import numpy as np

from config import N


def dropCost(DR_p, tasks):
    return np.sum(tasks[4, DR_p], -1)


def delayCost(T, Tasks):
    time = 0
    cost = 0
    for i in range(0, len(T)):
        tk = Tasks[:, T[i, :]]
        exe_time = max(tk[0], time)
        time = exe_time + tk[2]
        cost = cost + (exe_time - tk[0]) * tk[3] * (exe_time <= (tk[0] + tk[1]))
        cost = cost + tk[4] * (exe_time > (tk[0] + tk[1]))
    return cost


# [height, width], [size] ([0 1 ..0, 1 0 ..., ])
def scheduleCost(tasks, schedule):
    drop_len = len(schedule[schedule == N + 1])
    idx = np.argsort(schedule)
    drop_cost = dropCost(idx[N - drop_len + 1:N], tasks)
    delay_cost = delayCost(idx[1:N - drop_len], tasks)
    return drop_cost + delay_cost


def predictCost(tasks, predict):
    schedule = np.zeros([N + 1])
    for i in range(0, N + 1):
        schedule[i] = np.argmax(predict[i:N + 2:-1])

    return scheduleCost(tasks, schedule)
