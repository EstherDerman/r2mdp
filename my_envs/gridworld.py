import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape, p):
        # if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
        #     raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        Pmat = np.zeros([nS, nS, nA])
        Rmat = np.zeros([nS, nA])
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}  # P[s][a] = (prob, next_state, reward, is_done)

            is_done_l = lambda s: s == 0
            is_done_r = lambda s: s == (MAX_X - 1)
            is_done = is_done_l or is_done_r

            if is_done_l(s):
                reward = 1.0
            elif is_done_r(s):
                reward = 10.0
            else:
                reward = 0.0

            if is_done(s):  # terminal state
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:  # not a terminal state
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_left = s if x == 0 else s - 1
                ns_up = s if y == 0 else s - MAX_X
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X

                rest_right = ((1 - p) / 4, ns_right, reward, is_done(ns_right))
                rest_left = ((1 - p) / 4, ns_left, reward, is_done(ns_left))
                rest_up = ((1 - p) / 4, ns_up, reward, is_done(ns_up))
                rest_down = ((1 - p) / 4, ns_down, reward, is_done(ns_down))

                # P[s][a] = (prob, next_state, reward, is_done)
                P[s][RIGHT] = [((p + (1 - p) / 4), ns_right, reward, is_done(ns_right)), rest_down, rest_left, rest_up]
                P[s][LEFT] = [((p + (1 - p) / 4), ns_left, reward, is_done(ns_left)), rest_up, rest_right, rest_down]
                P[s][UP] = [((p + (1 - p) / 4), ns_up, reward, is_done(ns_up)), rest_right, rest_down, rest_left]
                P[s][DOWN] = [((p + (1 - p) / 4), ns_down, reward, is_done(ns_down)), rest_left, rest_up, rest_right]

            for a in range(nA):
                for prob, next_state, reward, done in P[s][a]:
                    Pmat[s][next_state][a] += prob
                Rmat[s][a] = reward

            it.iternext()

        isd = np.ones(nS) / nS  # uniform initial state distribution

        self.P = P
        self.Pmat = Pmat
        self.Rmat = Rmat
        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()