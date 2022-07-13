# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from collections import deque


class TimeLine:
    def __init__(self, T=None):
        self.taskIndices = np.array([], dtype='float16') if T is None else T.taskIndices
        self.delayCost = 0 if T is None else T.delayCost
        self.time = 0 if T is None else T.time

    def addTask(self, task_idx, tasks):
        self.taskIndices = np.append(self.taskIndices, task_idx)
        task = tasks[task_idx, :]
        execution_time = max(task[1], self.time)
        self.time = execution_time + task[3]
        self.delayCost = self.delayCost + (execution_time - task[1]) * task[4]


class Node:
    def __init__(self, T=None, PF=None, NS=None, DR=None):
        self.T = T if T is not None else TimeLine()
        self.PF = PF if PF is not None else np.array([], dtype='float16')
        self.NS = NS if NS is not None else np.array([], dtype='float16')
        self.DR = DR if DR is not None else np.array([], dtype='float16')


def BAB(tasks):
    stack = deque()
    UB = float('inf')
    n = Node()
    n.PF = deque(np.arange(0, tasks.shape[1]))
    stack.append(n)
    while stack.count != 0:
        n = stack[-1]
        if len(n.PF) != 0:
            j = n.PF[-1]
            n.PF.pop()
            T_p = TimeLine(n.T)
            PF_p = deque(set(n.PF + n.NS))
            NS_p = deque()
            n.NS = deque(set(n.NS + deque([j])))

            dropped_idx = (tasks[1, PF_p] + tasks[2, PF_p]) < T_p.time
            DR_p = deque(n.DR + PF_p[dropped_idx])
        pass


'''


while (stack.size ~= 0)
    n = stack.top();
    if(~isempty(n.PF))
        j = n.PF(end);
        n.PF(end)=[];
        T_p = n.T.deepcopy();
        T_p.addTask(j,tasks);
        PF_p = [n.PF, n.NS];
        PF_p = unique(PF_p);
        NS_p = [];
        n.NS = unique([n.NS, j]);
        
        dropped_idx = (tasks(1,PF_p)+tasks(2,PF_p)) < T_p.time;
        DR_p = [n.DR,PF_p(dropped_idx)];
        PF_p(dropped_idx) = [];

        C_p = T_p.delay_cost + droppingCost(DR_p, tasks);

        b1 = isActive(T_p, n.T, PF_p, tasks);
        b2 = isLOWSActive(T_p,tasks);
        if( b1 && b2&& C_p < UB)
            n_t =  Node();
            n_t.init(T_p, PF_p, NS_p, DR_p)
            stack.add(n_t);
            node_add = node_add + 1;
        end
        
    else
        drc =  droppingCost(n.DR,tasks);
        C = n.T.delay_cost + drc;
        if(isempty(n.NS) && C < UB)
            UB = C;
            T_optimal.schedule = n.T.task_indices;
            T_optimal.delay_cost = n.T.delay_cost;
            T_optimal.time = n.T.time;
            T_optimal.drop_cost = drc;
            T_optimal.node_add = node_add;
            T_optimal.node_rm = node_rm;
        end
        stack.remove();
        node_rm = node_rm + 1;
    end
end
end

function c = droppingCost(DR_p, tasks)
c = sum(tasks(5,DR_p));
end

function b = isActive(T_p, T, PF_p, tasks)
b = true;
added_task_idx = T_p.task_indices(end);
added_task = tasks(:, added_task_idx);
origin_time = T.time;
for i = 1 : length(PF_p)
    idx = PF_p(i);
    task = tasks(:,idx);
    t = max(task(1),origin_time) + task(3);
    t = max(added_task(1), t) + added_task(3);
    b = b && (t > T_p.time);
end
end

function b = isLOWSActive(T_p,tasks)
len = length(T_p.task_indices);
if len == 1
    b = true;
    return;
end
Ts = repmat(T_p.task_indices',1,len-1);
for i = 1 : len-1
    Ts(i:i+1,i) = Ts(i+1:-1:i,i);
end
c = computeDelayCost(Ts,tasks);
b = all(c >= T_p.delay_cost);
end

function costs = computeDelayCost(Ts,tasks)
len = size(Ts,2);
times = zeros(1,len);
costs = zeros(1,len);
for i = 1 : size(Ts,1)
    tks = tasks(:,Ts(i,:));
    exe_times = max(tks(1,:),times);
    times = exe_times + tks(3,:);
    costs = costs + (exe_times - tks(1,:)).*tks(4,:).*...
        (exe_times <= (tks(1,:)+tks(2,:)));
    costs = costs + tks(5,:).*(exe_times > (tks(1,:)+tks(2,:)));
end
end
'''


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
