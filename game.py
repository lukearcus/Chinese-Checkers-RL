import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import players

matplotlib.use("TkAgg")


class ChineseCheckers:
    """
    Chinese Checkers game
    """

    def __init__(self, _players, _draw=False, shuffle=True):
        for player in _players:
            if isinstance(player, players.human_player):
                _draw = True
                break
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
            self.fig.canvas.manager.full_screen_toggle()
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
        # if self.move_chosen != [0, 0]:
        #    RGB[self.move_chosen[0][0], self.move_chosen[0][1]][3] = 75
        self.ax.clear()
        self.ax.imshow(RGB)
        for i in range(self.num_players):
            self.ax.text(2*(i+1 > 3), i % 3, 'player ' + str(i+1),
                         bbox={
                              'facecolor': palette[self.players[i].id_num]/255,
                              'pad': 10})
        if self.win_order:
            self.ax.text(18, 0, 'Leaderboard',
                         bbox={'facecolor': 'white',
                               'pad': 10})
        for i, winner in enumerate(self.win_order):
            self.ax.text(20+2*(i+1 > 3), i % 3,
                         str(i+1) + ': ' + 'player ' + str(winner),
                         bbox={
                              'facecolor': palette[self.players[i].id_num]/255,
                              'pad': 10})

        # plt.title("Player: " + str(self.curr_player))
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
        not_empty = [True] * 6
        have_reached = [False] * 6
        for i in range(4):
            first_goal = self.board[-1-i, :]
            not_empty[0] = not_empty[0] and np.all(first_goal != 0)
            have_reached[0] = have_reached[0] or np.any(first_goal == 1)

            second_goal = self.board[-i-5, 0:9-i]
            not_empty[1] = not_empty[1] and np.all(second_goal != 0)
            have_reached[1] = have_reached[1] or np.any(second_goal == 2)

            third_goal = self.board[i+4, 0:9-i]
            not_empty[2] = not_empty[2] and np.all(third_goal != 0)
            have_reached[2] = have_reached[2] or np.any(third_goal == 3)

            fourth_goal = self.board[i, :]
            not_empty[3] = not_empty[3] and np.all(fourth_goal != 0)
            have_reached[3] = have_reached[3] or np.any(fourth_goal == 4)

            fifth_goal = self.board[i+4, -9+i:]
            not_empty[4] = not_empty[4] and np.all(fifth_goal != 0)
            have_reached[4] = have_reached[4] or np.any(fifth_goal == 5)

            sixth_goal = self.board[-i-5, -9+i:]
            not_empty[5] = not_empty[5] and np.all(sixth_goal != 0)
            have_reached[5] = have_reached[5] or np.any(sixth_goal == 6)

        won = [not_empty_i and reached_i for not_empty_i, reached_i
               in zip(not_empty, have_reached)]

        for player in self.players:
            if won[player.id_num - 1]:
                if player.id_num not in self.win_order:
                    self.win_order.append(player.id_num)
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

    def play_turn(self, curr_player, turn):
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
        self.board[selected_move[1][0], selected_move[1][1]] =\
            curr_player.id_num

        self.check_win(turn)
        if self.draw:
            self.show()

    def play_game(self):
        turn = 0
        while len(self.win_order) < self.num_players-1:
            for curr_player in self.players:
                if curr_player.id_num not in self.win_order:
                    self.play_turn(curr_player, turn)
            turn += 1
        self.win_order.append(self.players[0].id_num)
        self.win_turn.append(turn)

    def training(self, trainer):
        turn = 0
        while len(self.win_order) < self.num_players-1:
            old_hist = list()
            new_hist = list()
            for curr_player in self.players:
                old_hist.append(self.board)
                if curr_player.id_num not in self.win_order:
                    self.play_turn(curr_player, turn)
                new_hist.append(self.board)

            reward = [0]*6
            reward_values = np.linspace(-1, 1, self.num_players)
            for i, player in enumerate(self.win_order):
                if self.win_turn.count(self.win_turn[i]) > 1:
                    poses = [j for j, x in enumerate(self.win_turn)
                             if x == self.win_turn[i]]
                    reward[player-1] = (reward_values[poses[0]]
                                        + reward_values[poses[-1]])/2
                else:
                    reward[player-1] = reward_values[i]
            for i, player in enumerate(self.players):
                trainer.buffer.add(old_hist[i], new_hist[i],
                                   reward[player.id_num],
                                   not reward[player.id_num],
                                   player.id_num)
            turn += 1
            trainer.train_nn()
        self.win_order.append(self.players[0].id_num)
        self.win_turn.append(turn)
