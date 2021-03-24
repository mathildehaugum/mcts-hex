import collections
import random
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque()
        self.max_size = 1000

    def clear_buffer(self):
        """ Empty the buffer, which is performed in the beginning of the algorithm"""
        self.buffer = collections.deque()

    def add_case(self, root, D):
        """Add a new case to the buffer consisting of the root and the distribution of the
        visit counts in MCTS along all edges from the given root"""
        new_case = (root, D)  # D = distribution of visit counts in MCTS
        self.buffer.append(new_case)
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        print("Added: " + str((root, D)), " to buffer of length: " + str(len(self.buffer)))

    def get_random_minibatch(self):
        """Return a random minibatch of cases of the given size used to train the ANET (i.e. actor).
        If the batch size is larger than the buffer, the entire buffer is returned. If it is smaller,
        the elements in the buffer is weighted depending on when they where added to the buffer and
        a random batch is chosen depending on these weights"""
        batch_size = 10
        if len(self.buffer) < batch_size:
            return self.buffer
        else:
            buffer_weights = np.linspace(0.0, 1.0, len(self.buffer))  # Newer cases > Older cases
            weighted_batch = random.choices(population=self.buffer, weights=buffer_weights, k=batch_size)
            return weighted_batch






