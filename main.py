import players
import game
import value_nets

network = value_nets.NeuralNetwork_v1().to('cuda')
RL_player = players.RL_player_v1(network)
player_list = [RL_player, players.deterministic_player()]
a = game.ChineseCheckers(player_list, 1)
a.play_game()
print(a.win_order)
print(a.win_turn)
#test
