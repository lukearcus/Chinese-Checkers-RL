import players
import game

player_list = [players.random_player(), players.deterministic_player()]
a = game.ChineseCheckers(player_list)
a.play_game()
print(a.win_order)

#test
