import players
import game
import value_nets
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = value_nets.NeuralNetwork_v1().to(device)
RL_player = players.RL_player_v1(network)
player_list = [RL_player, players.deterministic_player()]

draw = False
board = game.ChineseCheckers(player_list, draw)
for itt in range(100):
    board.play_game()
    reward_values = np.linspace(-1, 1, len(player_list))
    reward = [0]*6
    for i, player in enumerate(board.win_order):
        if board.win_turn.count(board.win_turn[i]) > 1:
            poses = [j for j, x in enumerate(board.win_turn) if x == board.win_turn[i]]
            reward[player-1] = (reward_values[poses[0]] + reward_values[poses[-1]])/2
        else:
            reward[player-1] = reward_values[i]
    board.reset(player_list, draw)
    for player in player_list:
        if issubclass(type(player), players.RL_Player):
            player.train(reward)
            player.inc_iter()


print(reward)
