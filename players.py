import random
import numpy as np


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
        return random.choice(poss_moves)


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
        return selected_move


class RL_player_v1(Player):
    def __init__(self, val_nn):
        self.value_network = val_nn

    def get_val(self, board):

        values = self.value_network(board)
        value = values[self.id_num]
        return value

    def move(self, board, poss_moves):
        max_val = -np.inf
        for move in poss_moves:
            poss_board = np.copy(board)
            poss_board[move[0][0], move[0][1]] = 0
            poss_board[move[1][0], move[1][1]] = self.id_num
            value = self.get_val(poss_board)
            if value > max_val:
                max_val = value
                selected_move = move
        return selected_move
