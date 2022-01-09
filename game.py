import numpy as np
import players
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random

class ChineseCheckers:
    """
    Chinese Checkers game
    """

    def __init__(self, _players, _draw=False, shuffle=True):
        self.reset(_players, _draw, shuffle)

    def reset(self, _players, _draw, shuffle):
        plt.ion()
        NewBoard = np.zeros((17, 27))
        num_players = len(_players)
        self.players = _players
        if shuffle:
            random.shuffle(self.players)

        self.win_order = list()
        self.win_turn = list()
        if num_players == 2:
            self.players[0].set_goal(np.array([16, 13]))
            self.players[0].set_id(1)
            self.players[1].set_goal(np.array([0, 13]))
            self.players[1].set_id(4)
            for i in range(4):
                NewBoard[i, :] = 1
                NewBoard[-1-i, :] = 4
        if num_players == 3:
            self.players[0].set_goal(np.array([16, 13]))
            self.players[0].set_id(1)
            self.players[1].set_goal(np.array([4, 1]))
            self.players[1].set_id(3)
            self.players[2].set_goal(np.array([4, 25]))
            self.players[2].set_id(5)
            for i in range(4):
                NewBoard[i, :] = 1
                NewBoard[-i-5, -9+i:] = 3
                NewBoard[-i-5, 0:9-i] = 5
        if num_players == 4:
            self.players[0].set_goal(np.array([12, 1]))
            self.players[0].set_id(2)
            self.players[1].set_goal(np.array([4, 1]))
            self.players[1].set_id(3)
            self.players[2].set_goal(np.array([4, 25]))
            self.players[2].set_id(5)
            self.players[3].set_goal(np.array([12, 25]))
            self.players[3].set_id(6)
            for i in range(4):
                NewBoard[i+4, -9+i:] = 2
                NewBoard[-i-5, -9+i:] = 3
                NewBoard[-i-5, 0:9-i] = 5
                NewBoard[i+4, 0:9-i] = 6
        if num_players == 6:
            self.players[0].set_goal(np.array([16, 13]))
            self.players[0].set_id(1)
            self.players[1].set_goal(np.array([12, 1]))
            self.players[1].set_id(2)
            self.players[2].set_goal(np.array([4, 1]))
            self.players[2].set_id(3)
            self.players[3].set_goal(np.array([0, 13]))
            self.players[3].set_id(4)
            self.players[4].set_goal(np.array([4, 25]))
            self.players[4].set_id(5)
            self.players[5].set_goal(np.array([12, 25]))
            self.players[5].set_id(6)
            for i in range(4):
                NewBoard[i, :] = 1
                NewBoard[i+4, -9+i:] = 2
                NewBoard[-i-5, -9+i:] = 3
                NewBoard[-1-i, :] = 4
                NewBoard[-i-5, 0:9-i] = 5
                NewBoard[i+4, 0:9-i] = 6

        for i in range(4):
            for j in range(int((NewBoard.shape[1]-1)/2)-i):
                NewBoard[i, j] = -1
                NewBoard[i, -1-j] = -1
                NewBoard[-1-i, j] = -1
                NewBoard[-1-i, -1-j] = -1
            NewBoard[i+5, 0:i+1] = -1
            NewBoard[i+5, -i-1:] = -1
            NewBoard[-i-6, 0:i+1] = -1
            NewBoard[-i-6, -i-1:] = -1

        for i in range(NewBoard.shape[0]):
            for j in range(NewBoard.shape[1]):
                if j % 2 == 0:
                    if i % 2 == 0:
                        NewBoard[i, j] = -1
                else:
                    if i % 2 != 0:
                        NewBoard[i, j] = -1

        self.allowed = np.zeros(NewBoard.shape).astype(bool)
        self.move_chosen = [0, 0]
        self.board = NewBoard

        self.num_players = num_players
        self.win_list = [False]*num_players
        self.draw = _draw
        if _draw:
            self.fig = plt.figure()
            self.fig.canvas.manager.full_screen_toggle()  # toggle fullscreen mode
            self.ax = self.fig.add_subplot(111)
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.show()
    def show(self):
        palette = np.array([[255,   255,   255, 255],   # black
                            [255,   0,   0, 255],   # red
                            [0, 255,   0, 255],   # green
                            [0,   0, 255, 255],   # blue
                            [0, 255, 255, 255],
                            [255, 0, 255, 255],
                            [255, 255, 0, 255],
                            [0, 0, 0, 255]])  # white
        RGB = palette[(self.board).astype(int)]
        #if self.move_chosen != [0, 0]:
        #    RGB[self.move_chosen[0][0], self.move_chosen[0][1]][3] = 75
        self.ax.clear()
        self.ax.imshow(RGB)
        for i in range(self.num_players):
            self.ax.text(2*(i+1 > 3), i % 3, 'player ' + str(i+1),
                         bbox={'facecolor': palette[self.players[i].id_num]/255, 'pad': 10})
        if self.win_order:
            self.ax.text(18, 0, 'Leaderboard', bbox={'facecolor': 'white', 'pad': 10})
        for i, winner in enumerate(self.win_order):
            self.ax.text(20+2*(i+1 > 3), i % 3, str(i+1) + ': ' +  'player ' + str(winner),
                         bbox={'facecolor': palette[self.players[i].id_num]/255, 'pad': 10})

        #plt.title("Player: " + str(self.curr_player))
        plt.show()

    def check_allowed(self, pos, hopping):
        check = list()
        if pos[1] < 25:
            check.append((pos[0], pos[1]+2))
        if pos[1] > 2:
            check.append((pos[0], pos[1]-2))
        if pos[0] < 16:
            if pos[1] < 26:
                check.append((pos[0]+1, pos[1]+1))
            if pos[1] > 0:
                check.append((pos[0]+1, pos[1]-1))
        if pos[0] > 0:
            if pos[1] < 26:
                check.append((pos[0]-1, pos[1]+1))
            if pos[1] > 0:
                check.append((pos[0]-1, pos[1]-1))

        for elem in check:
            if not(self.board[elem]):
                self.allowed[elem] = True and not(hopping)
            else:
                hop = (pos[0]+2*(elem[0]-pos[0]), pos[1]+2*(elem[1]-pos[1]))
                in_range = hop[0] < 17 and hop[0] >= 0
                in_range = in_range and hop[1] < 27
                in_range = in_range and hop[1] >= 0
                if in_range:
                    if not(self.board[hop]) and not(self.allowed[hop]):
                        self.allowed[hop] = True
                        self.check_allowed(hop, True)

    def check_win(self, turn):
        won = list()
        for i in range(self.num_players):
            won.append(True)
        if self.num_players == 2:
            for i in range(4):
                won[0] = won[0] and (np.any(self.board[-1-i, :] == 1)
                                     and np.all(self.board[-1-i, :] != 0))
                won[1] = won[1] and (np.any(self.board[i, :] == 4)
                                     and np.all(self.board[i, :] != 0))
        if self.num_players == 3:
            for i in range(4):
                won[0] = won[0] and np.any(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 3)
                won[2] = won[2] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 5)
        if self.num_players == 4:
            for i in range(4):
                won[0] = won[0] and np.any(self.board[-i-5, np.pad(self.board[-i-5, 0:9-i] != -1, (0, i+18))] == 2)
                won[1] = won[1] and np.any(self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 3)
                won[2] = won[2] and np.any(self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 5)
                won[3] = won[3] and np.any(self.board[-i-5, np.pad(self.board[-i-5, -9+i:] != -1, (18+i, 0))] == 6)
        if self.num_players == 6:
            for i in range(4):
                won[0] = won[0] and np.any(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.any(
                        self.board[-i-5, np.pad(self.board[-i-5, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 3)
                won[3] = won[3] and np.any(self.board[i, self.board[i, :] != -1] == 4)
                won[4] = won[4] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0) )] == 5)
                won[5] = won[5] and np.any(
                        self.board[-i-5, np.pad(self.board[-i-5, -9+i:] != -1, (18+i, 0))] == 6)
        self.win_list = won
        for i in range(self.num_players):
            if won[i]:
                if self.players[i].id_num not in self.win_order:
                    self.win_order.append(self.players[i].id_num)
                    self.win_turn.append(turn)
                    

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        ix = round(ix)
        iy = round(iy)
        pos = self.board[(iy, ix)]

        if pos:
            self.move_chosen[0] = np.array((iy, ix))
        else:
            self.move_chosen[1] = np.array((iy, ix))

    def play_game(self, buffer):
        turn = 0
        while len(self.win_order) < self.num_players-1:
            for i, curr_player in enumerate(self.players):
                if curr_player.id_num not in self.win_order:
                    positions = np.where(self.board == curr_player.id_num)
                    possible_moves = []
                    positions = np.stack(positions).T
                    for pos in positions:
                        self.allowed[:, :] = False
                        self.check_allowed(pos, False)
                        allowed_moves = np.where(self.allowed)
                        allowed_moves = np.stack(allowed_moves).T
                        for move in allowed_moves:
                            possible_moves.append((pos, move))
                    if isinstance(curr_player, players.human_player):
                        while True:
                            self.fig.canvas.get_tk_widget().update()
                            for possible in possible_moves:
                                if np.array_equal(tuple(self.move_chosen), possible):
                                    selected_move = tuple(self.move_chosen)
                                    break
                            else:
                                continue
                            break
                        self.move_chosen[0] = 0
                        self.move_chosen[1] = 0
                    else:
                        selected_move = curr_player.move(self.board, possible_moves)
                        if self.draw:
                            plt.waitforbuttonpress()
                    self.board[selected_move[0][0], selected_move[0][1]] = 0
                    self.board[selected_move[1][0], selected_move[1][1]] = curr_player.id_num
                    self.check_win(turn)
                    if self.draw:
                        self.show()
            reward = [0]*6
            reward_values = np.linspace(-1, 1, self.num_players)
            for i, player in enumerate(self.win_order):
                if self.win_turn.count(self.win_turn[i]) > 1:
                    poses = [j for j, x in enumerate(self.win_turn) if x == self.win_turn[i]]
                    reward[player-1] = (reward_values[poses[0]] + reward_values[poses[-1]])/2
                else:
                    reward[player-1] = reward_values[i]
            for player in self.players:
                if not isinstance(player, players.human_player):
                    buffer.add(player.old_history, player.new_history,
                               reward[player.id_num], not reward[player.id_num],
                               player.id_num) 
            turn += 1
        self.win_order.append(self.players[0].id_num)
        self.win_turn.append(turn)
