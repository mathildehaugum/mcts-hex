"""Use NN to update target/default policy. Takes board state as input and produces
a probability distribution over all possible moves from that state as output. Aim is
to produce an intelligent target policy that can later be used independently of MCTS as an actor module"""
from math import floor, sqrt
import tensorflow as tf
import numpy as np
from tensorflow import keras as KER
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import zeros_like
import random
from matplotlib import pyplot as plt


class Actor:
    def __init__(self, learning_rate, epsilon, decay_rate, board_size, nn_dims, activation, optimizer, loss_function, save_dir=None, load_dir=None):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.size = board_size
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.anet = ANET(nn_dims, activation, optimizer, loss_function, learning_rate)
        self.actor_loss = []

    def target_policy(self, state, player, is_top_policy=False):
        if not is_top_policy and self.epsilon >= random.uniform(0, 1):
            legal_indexes = []
            for index in range(len(state)):
                if state[index] == (0, 0):
                    legal_indexes.append(index)
            action_index = legal_indexes[random.randrange(len(legal_indexes))]
        else:
            tensor_state = convert_state_to_tensor(state, player)
            prediction = self.anet.predict(tensor_state)
            for index in range(len(state)):
                if state[index] == (0, 0):
                    prediction[index] = 0
            total = np.sum(prediction)
            prediction *= 1/total  # re-normalize after non-legal action indexes are set to 0
            action_index = prediction[max(prediction)]
        return self.get_action_from_index(action_index, state, player)

    @staticmethod
    def get_action_from_index(action_index, state, player):
        cell_row = floor(action_index/sqrt(len(state)))
        cell_col = int(action_index % sqrt(len(state)))
        chosen_action = [(cell_row, cell_col), player]
        return chosen_action

    def train(self, minibatch):
        """Train neural network on given random minibatch from replay buffer. The minibatch contains several
         cases, where each case=(s,D) (i.e. state and its distribution produced by MCTS)"""
        # input to fit can be numpy array, so maybe try to get minibatch in this format?
        print("**************** ENTERING TRAINING ****************")
        print(minibatch)
        # loss = self.anet.model.train(x_train, y_train)
        # self.actor_loss.append(loss)
        # OBS: se split_gd for hjelp?

    def decay_epsilon(self):
        """Decay epsilon to reduce the amount of exploring in later episodes"""
        self.epsilon *= self.decay_rate

    def save(self, episode_num):
        """Save current parameter of ANET for later use in tournament play"""
        path = "./models/ANET_" + episode_num + "_size_" + self.size
        self.anet.model.save_weights(filepath=path)

    def load(self, path):
        """Load saved parameter of ANET for use in tournament play"""
        self.anet.model.load_weights(filepath=path)

    def plot_loss_and_accuracy(self):  # OBSOBS: ADD ACCURACY?
        """Print loss and accuracy of all episodes"""
        episodes = list(range(1, len(self.actor_loss) + 1))
        plt.title("The average actor loss for each episode")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.plot(episodes, self.actor_loss)
        plt.savefig("./images/actor_loss.png")
        plt.show()


class ANET:
    def __init__(self, nn_dims, activation, optimizer, loss_function, learning_rate):
        self.input_size, self.hidden_layers_dim, self.output_size = nn_dims[0], nn_dims[1:len(nn_dims)-1], nn_dims[len(nn_dims)-1]

        self.activation = self.get_activation_function(activation)
        self.optimizer = self.get_optimizer(optimizer)
        self.loss_function = self.get_loss_function(loss_function)
        self.alpha = learning_rate
        self.model = self.init_nn()

    def init_nn(self):
        model = KER.models.Sequential()
        model.add(KER.layers.Dense(self.input_size, activation=self.activation, input_shape=(self.input_size, )))
        for i in range(len(self.hidden_layers_dim)):
            model.add(KER.layers.Dense(self.hidden_layers_dim[i], activation=self.activation))
        model.add(KER.layers.Dense(self.output_size))
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["mean_squared_error"])
        # model.summary()
        return model

    def predict(self, tensor_state):
        """Forward pass in neural model to produce distribution over action indexes"""
        return self.model(tensor_state)

    def train(self, x_train, y_train):  # backward-pass in neural network
        history = self.model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=True, callbacks=[])
        return history.history['loss']


    @staticmethod
    def get_activation_function(activation):
        if activation == "sigmoid":
            return None
        elif activation == "relu":
            return None
        elif activation == "linear":
            return None
        elif activation == "tanh":
            return None

    @staticmethod
    def get_optimizer(optimizer):
        if optimizer == "adam":
            return None
        elif optimizer == "adagrad":
            return None
        elif optimizer == "sgd":
            return None
        elif optimizer == "adadelta":
            return None

    @staticmethod
    def get_loss_function(loss):
        if loss == "mean-squared-error":
            return None
        elif loss == "cross-entropy":
            return None


def convert_state_to_tensor(state):
    """Convert given state from string format to tensor (i.e. array-like object)"""
    state_array = np.array(list(state))  # make numpy array of string
    tensor_state = tf.convert_to_tensor(state_array, np.float32)  # convert numpy array to tensor
    return tf.convert_to_tensor(np.expand_dims(tensor_state, axis=0))  # insert axis on pos 0 to get a tensor shape that corresponds to the input_shape of the neural network (e.g. (15, ) --> (1, 15))
