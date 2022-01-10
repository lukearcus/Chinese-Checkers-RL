import random
import numpy as np
import torch

# Receives board position and moves as a list of tuples (position (of piece to
# be moved), position it can move to)


class Player:
    id_num = 0

    def move(self, board, poss_moves):
        return None

    def set_goal(self, goal):
        return None

    def set_id(self, id_num):
        self.id_num = id_num


class human_player(Player):
    def move(self, poss_moves):
        print(poss_moves)
        x1 = int(input())
        y1 = int(input())
        x2 = int(input())
        y2 = int(input())
        return (np.array((x1, y1)), np.array((x2, y2)))


class random_player(Player):
    def move(self, board, poss_moves):
        move = random.choice(poss_moves)
        selected_board = np.copy(board)
        selected_board[move[0][0], move[0][1]] = 0
        selected_board[move[1][0], move[1][1]] = self.id_num
        return move


class deterministic_player(Player):
    rand_freq = 0.5
    goal_pos = (0, 0)

    def set_goal(self, goal):
        self.goal_pos = goal

    def move(self, board, poss_moves):
        if random.random() < self.rand_freq:
            selected_move = random.choice(poss_moves)
        else:
            positions = np.where(board == self.id_num)
            positions = np.stack(positions).T
            min_distance = np.inf
            for move in poss_moves:
                dist = 0
                for pos in positions:
                    if pos[0] == move[0][0] and pos[1] == move[0][1]:
                        dist += np.linalg.norm(self.goal_pos-move[1])**2
                    else:
                        dist += np.linalg.norm(self.goal_pos-pos)**2
                if dist < min_distance:
                    min_distance = dist
                    selected_move = move
        selected_board = np.copy(board)
        selected_board[selected_move[0][0], selected_move[0][1]] = 0
        selected_board[selected_move[1][0], selected_move[1][1]] = self.id_num
        return selected_move


class RL_Player(Player):
    def __init__(self):
        return None

    def get_val(self):
        return None

    def train(self):
        return None

    def inc_iter(self):
        return None

    def reset(self):
        return None


class RL_player_v1(RL_Player):

    def __init__(self, val_nn, eps=0.3, _gamma=0.9, learning_rate=1e-3):
        self.value_network = val_nn
        self.iter = 0
        self.epsilon = eps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_val(self, board):
        x = torch.from_numpy(board).float().to(self.device)
        x = x.flatten()
        x = x.unsqueeze(0)
        values = self.value_network(x)
        value = values[0, self.id_num]
        return value

    def move(self, board, poss_moves):
        if random.random() < self.epsilon*np.exp(-self.iter/100):
            selected_move = random.choice(poss_moves)
            selected_board = np.copy(board)
            selected_board[selected_move[0][0], selected_move[0][1]] = 0
            selected_board[selected_move[1][0], selected_move[1][1]] =\
                self.id_num
        else:
            max_val = -np.inf
            for move in poss_moves:
                poss_board = np.copy(board)
                poss_board[move[0][0], move[0][1]] = 0
                poss_board[move[1][0], move[1][1]] = self.id_num
                value = self.get_val(poss_board)
                if value > max_val:
                    max_val = value
                    selected_move = move
                    selected_board = poss_board
        return selected_move

    def inc_iter(self):
        self.iter += 1
        return
