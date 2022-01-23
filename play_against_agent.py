import players
import game
import value_nets
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = value_nets.NeuralNetwork_v1().to(device)

network.load_state_dict(torch.load('trained_NN_v1_1_weights'))
trainer = value_nets.NN_Trainer(network)
RL_player = players.RL_player_v1(network, 0)
player_list = [players.human_player(), RL_player]

shuffle = False
board = game.ChineseCheckers(player_list, shuffle=shuffle)
board.play_game()
print("Winner was player " +
      str(board.win_order[0]) + ', reached turn ' +
      str(board.win_turn[-1]))
