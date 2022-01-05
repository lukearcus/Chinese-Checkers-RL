import numpy as np
import players
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class ChineseCheckers:
    """
    Chinese Checkers game
    """

    win_order = list()

    def __init__(self, _players, _draw=False):
        plt.ion()
        NewBoard = np.zeros((17, 27))
        num_players = len(_players)
        self.players = _players
        for i, player in enumerate(self.players):
            player.set_id(i+1)
        if num_players == 2:
            self.players[0].set_goal(np.array([16, 13]))
            self.players[1].set_goal(np.array([0, 13]))
            for i in range(4):
                NewBoard[i, :] = 1
                NewBoard[-1-i, :] = 2
        if num_players == 3:
            for i in range(4):
                NewBoard[i, :] = 1
                NewBoard[-i-5, -9+i:] = 2
                NewBoard[-i-5, 0:9-i] = 3
        if num_players == 4:
            for i in range(4):
                NewBoard[i+4, -9+i:] = 1
                NewBoard[-i-5, -9+i:] = 2
                NewBoard[-i-5, 0:9-i] = 3
                NewBoard[i+4, 0:9-i] = 4
        if num_players == 6:
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
                         bbox={'facecolor': palette[i+1]/255, 'pad': 10})
        if self.win_order:
            self.ax.text(18, 0, 'Leaderboard', bbox={'facecolor': 'white', 'pad': 10})
        for i, winner in enumerate(self.win_order):
            self.ax.text(20+2*(i+1 > 3), i % 3, str(i+1) + ': ' +  'player ' + str(winner),
                         bbox={'facecolor': palette[winner]/255, 'pad': 10})

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

    def check_win(self):
        won = list()
        for i in range(self.num_players):
            won.append(True)
        if self.num_players == 2:
            for i in range(4):
                won[0] = won[0] and (np.any(self.board[-1-i, :] == 1)
                                     and np.all(self.board[-1-i, :] != 0))
                won[1] = won[1] and (np.any(self.board[i, :] == 2)
                                     and np.all(self.board[i, :] != 0))
        if self.num_players == 3:
            for i in range(4):
                won[0] = won[0] and np.any(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.any(
                        self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 3)
        if self.num_players == 4:
            for i in range(4):
                won[0] = won[0] and np.any(self.board[-i-5, np.pad(self.board[-i-5, 0:9-i] != -1, (0, i+18))] == 1)
                won[1] = won[1] and np.any(self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.any(self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 3)
                won[3] = won[3] and np.any(self.board[-i-5, np.pad(self.board[-i-5, -9+i:] != -1, (18+i, 0))] == 4)
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
                if i+1 not in self.win_order:
                    self.win_order.append(i+1)
                    self.players.pop(i)
                    

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        ix = round(ix)
        iy = round(iy)
        pos = self.board[(iy, ix)]

        if pos:
            self.move_chosen[0] = np.array((iy, ix))
        else:
            self.move_chosen[1] = np.array((iy, ix))

    def play_game(self):
        itt = 0
        while len(self.players) > 1:
            for i, curr_player in enumerate(self.players):
                if curr_player.id_num not in self.win_order:
                    positions = np.where(self.board == i+1)
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
                    self.board[selected_move[1][0], selected_move[1][1]] = i+1
                    self.check_win()
                    if self.draw:
                        self.show()
                    itt += 1
        self.win_order.append(self.players[0].id_num)
