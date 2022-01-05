import random  # in order to randomly choose starting player
import numpy as np
import matplotlib.pyplot as plt


class ChineseCheckers:
    """
    Chinese Checkers game
    """

    win_order = list()

    def __init__(self, num_players, _draw=False):
        NewBoard = np.zeros((17, 27))
        
        if num_players == 2:
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
        self.chosen = (0, 0)
        self.board = NewBoard
        self.curr_player = random.randint(1, num_players)
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
        if self.chosen != (0, 0):
            RGB[self.chosen][3] = 75
        RGB[self.allowed] = [0, 0, 0, 75]
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

        plt.title("Player: " + str(self.curr_player))
        plt.show()

    def check_allowed(self, pos, hopping):
        check = list()
        if pos[1] < 25:
            check.append((pos[0], pos[1]+2))
        if pos[1] > 2:
            check.append((pos[0], pos[1]-2))
        if pos[0] < 16:
            check.append((pos[0]+1, pos[1]+1))
            check.append((pos[0]+1, pos[1]-1))
        if pos[0] > 1:
            check.append((pos[0]-1, pos[1]+1))
            check.append((pos[0]-1, pos[1]-1))

        for elem in check:
            if not(self.board[elem]):
                self.allowed[elem] = True and not(hopping)
            else:
                hop = (pos[0]+2*(elem[0]-pos[0]), pos[1]+2*(elem[1]-pos[1]))
                in_range = hop[0] < 17 and hop[0] >= 0
                in_range = in_range and hop[1] < 27
                in_range = in_range and hop[0] >= 0
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
                won[0] = won[0] and np.all(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.all(self.board[i, self.board[i, :] != -1] == 2)
        if self.num_players == 3:
            for i in range(4):
                won[0] = won[0] and np.all(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.all(
                        self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.all(
                        self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 3)
        if self.num_players == 4:
            for i in range(4):
                won[0] = won[0] and np.all(self.board[-i-5, np.pad(self.board[-i-5, 0:9-i] != -1, (0, i+18))] == 1)
                won[1] = won[1] and np.all(self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.all(self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0))] == 3)
                won[3] = won[3] and np.all(self.board[-i-5, np.pad(self.board[-i-5, -9+i:] != -1, (18+i, 0))] == 4)
        if self.num_players == 6:
            for i in range(4):
                won[0] = won[0] and np.all(self.board[-1-i, self.board[-1-i, :] != -1] == 1)
                won[1] = won[1] and np.all(
                        self.board[-i-5, np.pad(self.board[-i-5, 0:9-i] != -1, (0, i+18))] == 2)
                won[2] = won[2] and np.all(
                        self.board[i+4, np.pad(self.board[i+4, 0:9-i] != -1, (0, i+18))] == 3)
                won[3] = won[3] and np.all(self.board[i, self.board[i, :] != -1] == 4)
                won[4] = won[4] and np.all(
                        self.board[i+4, np.pad(self.board[i+4, -9+i:] != -1, (18+i, 0) )] == 5)
                won[5] = won[5] and np.all(
                        self.board[-i-5, np.pad(self.board[-i-5, -9+i:] != -1, (18+i, 0))] == 6)
        self.win_list = won
        for i in range(self.num_players):
            if won[i]:
                if i+1 not in self.win_order:
                    self.win_order.append(i+1)

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        ix = round(ix)
        iy = round(iy)
        pos = self.board[(iy, ix)]

        if pos == self.curr_player:
            self.chosen = (iy, ix)
            self.allowed[:, :] = False
            self.check_allowed(self.chosen, False)
        else:
            if self.allowed[(iy, ix)]:
                self.board[(iy, ix)] = self.curr_player
                self.board[self.chosen] = 0
                self.chosen = (0, 0)
                self.allowed[:, :] = False
                for i in range(self.curr_player+1, self.curr_player+self.num_players):
                    player = i
                    if player > self.num_players:
                        player -= self.num_players
                    if not(self.win_list[player-1]):
                        self.curr_player = player
                        break
        self.check_win()
        self.show()


plt.ion()
a = ChineseCheckers(2, True)
input()
