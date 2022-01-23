import players
import game
import value_nets
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = value_nets.NeuralNetwork_v2().to(device)

network.load_state_dict(torch.load('trained_NN_v2_2_weights'))
trainer = value_nets.NN_Trainer(network)
RL_player = players.RL_player_v1(network, 0)
player_list = [players.deterministic_player(), RL_player]

games_won = 0
shuffle = False
draw = True
board = game.ChineseCheckers(player_list, draw, shuffle)
for itt in range(100):
    print("Playing game " + str(itt) + "...")
    board.play_game()
    if board.win_order[0] == 1:
        games_won += 1
    board.reset(player_list, draw, shuffle)
print("RL agent won " + str(games_won/10) + " percent of games")
