import torch
from torch import nn
import random


class NeuralNetwork_v1(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v1, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(17*27, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 6),
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        vals = self.linear_relu_stack(x)
        return vals


class NeuralNetwork_v2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(17*27, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 6),
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        vals = self.linear_relu_stack(x)
        return vals


class ReplayBuffer:
    def __init__(self, _max=10000):
        self.history = []
        self.max_size = _max

    def add(self, init_state, next_state, reward, not_done, id_num):
        data = {
                "old": init_state,
                "new": next_state,
                "reward": reward,
                "not_done": not_done,
                "player_id": id_num
                }
        self.history.append(data)
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def get_batch(self, size):
        size = min(size, len(self.history))
        return random.sample(self.history, size)


class NN_Trainer:

    def __init__(self, nn, _batch_size=1000, _gamma=0.9, learning_rate=1e-3):
        self.buffer = ReplayBuffer()
        self.batch_size = _batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma = _gamma
        self.loss_fun = torch.nn.MSELoss()
        self.value_network = nn
        self.optimiser = torch.optim.RMSprop(self.value_network.parameters(),
                                             lr=learning_rate)

    def train_nn(self):
        batch = self.buffer.get_batch(self.batch_size-6)
        batch = batch + self.buffer.history[-6:]
        actual_batch_size = len(batch)
        new = torch.empty(actual_batch_size,
                          batch[0]["old"].size).to(self.device)
        old = torch.empty(actual_batch_size,
                          batch[0]["old"].size).to(self.device)
        old.require_grad = True
        reward = torch.zeros(actual_batch_size, 1).to(self.device)
        not_done = torch.zeros(actual_batch_size, 1).to(self.device)
        player = torch.zeros(actual_batch_size, 1).to(self.device)
        for i in range(actual_batch_size):
            old[i, :] = torch.from_numpy(batch[i]["old"]).float().flatten()
            new[i, :] = torch.from_numpy(batch[i]["new"]).float().flatten()
            reward[i] = batch[i]["reward"]
            not_done[i] = batch[i]["not_done"]
            player[i] = batch[i]["player_id"]-1
        ids = player.long()
        not_done = not_done.unsqueeze(0)
        new_vals = self.value_network(new)
        new_vals = new_vals.gather(1, ids.view(-1, 1))
        target = reward +\
            not_done*(self.gamma*new_vals)
        old_vals = self.value_network(old).gather(1, ids.view(-1, 1))
        target = target[0, :]
        loss = self.loss_fun(target.detach(), old_vals)
        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()
