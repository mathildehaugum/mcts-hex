import collections
import random
import numpy as np


class ReplayBuffer:
    def __init__(self):
        """ Class for making the replay buffer that contains the cases = (state, distribution) that are used
        to train the actor neural network (ANET). This buffer is filled with the result of each episode,
        and a random minibatch is created to be used in NN training"""
        self.buffer = collections.deque()
        self.max_size = 750  # restricted buffer size means that old cases are eventually replaced by newer cases that may give better results

    def clear_buffer(self):
        """ Empty the buffer, which is performed in the beginning of each RL episode"""
        self.buffer = collections.deque()

    def add_case(self, node, D):
        """Add a new case to the buffer consisting of the root node state with a player indicator and the
        normalized distribution of the visit counts found in MCTS along all edges from the given root"""
        player_board_state = [node.get_player()]
        player_board_state.extend(node.get_state())  # player indicator is added to ensure that visit distribution is attuned to the player (good moves for 1 is bad for 2, and vice versa)
        new_case = (player_board_state, D)
        self.buffer.append(new_case)
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        # print("Added: " + str((root, D)), " to buffer of length: " + str(len(self.buffer)))

    def get_random_minibatch(self):
        """Return a random minibatch of cases of the given size used to train the ANET (i.e. actor). If the batch size
        is larger than the buffer, the entire buffer is returned. If it is smaller, a random batch is selected"""
        batch_size = min(len(self.buffer), 256)  # prev: 64
        weights = np.linspace(0.0, 1.0, len(self.buffer))  # Add weights to cases, so newer cases are prioritized
        random_batch = random.choices(population=self.buffer, weights=weights, k=batch_size)
        return random_batch






