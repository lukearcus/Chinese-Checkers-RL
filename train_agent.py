import players
import game
import value_nets
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = value_nets.NeuralNetwork_v2().to(device)

network.load_state_dict(torch.load('trained_NN_v2_2_weights'))
trainer = value_nets.NN_Trainer(network)
RL_player = players.RL_player_v1(network)
player_list = [RL_player, players.deterministic_player()]
#RL_player_2 = players.RL_player_v1(network)
#player_list = [RL_player, RL_player_2]

shuffle = True
draw = False
board = game.ChineseCheckers(player_list, draw, shuffle)

try:
    for itt in range(1000):
        board.training(trainer)
        for player in player_list:
            if issubclass(type(player), players.RL_Player):
                player.inc_iter()
        print("Iteration " + str(itt) + ": Winner was player " +
              str(board.win_order[0]) + ', reached turn ' +
              str(board.win_turn[-1]))
        board.reset(player_list, draw, shuffle)
    torch.save(network.state_dict(), 'trained_NN_v2_2_weights')
except KeyboardInterrupt:
    print("Interrupted")
    torch.save(network.state_dict(), 'trained_NN_v2_2_weights')
