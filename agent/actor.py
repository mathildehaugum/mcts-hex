from math import floor, sqrt
import tensorflow as tf
import numpy as np
from tensorflow import keras as KER
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
import random


class Actor:
    """ Class for making an Actor that represents the NN that given a state will produce a probability
    distribution over all possible moves from that state. The network is trained in each episode
    to build an intelligent target policy that can be used in the tournament against other players"""
    def __init__(self, learning_rate, epsilon, decay_rate, board_size, nn_dims, activation, optimizer, loss_function, filename=""):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.size = board_size
        self.anet = ANET(board_size, nn_dims, activation, optimizer, loss_function, learning_rate)
        self.name = ""  # Used in tournament
        self.filename = filename  # Used in tournament

    def set_name(self, name):
        """ Sets name of actor used during the tournament to differ between the agents playing against each other"""
        self.name = name + "_" + self.filename.split("ep_")[1].split(".h5")[0]

    def target_policy(self, state, player, is_top_policy=False):
        """ The target/default policy (on-policy) that is used to choose actions during rollout simulations in MCTS.
        The random element ensures exploration during rollout, while during tournament is_top_policy = True to only use the NN (i.e. exploitation)"""
        if not is_top_policy and self.epsilon >= random.uniform(0, 1):
            legal_indexes = []
            for index in range(len(state)):
                if state[index] == 0:
                    legal_indexes.append(index)
            action_index = legal_indexes[random.randrange(len(legal_indexes))]
        else:
            player_board_state = [player]
            player_board_state.extend(state)
            tensor_state = convert_to_tensor(player_board_state)
            prediction = self.anet.predict(tensor_state).numpy()[0]  # Numpy array containing predicted desirability for each child state (i.e. actions leading to child states). Use of 0: [[...]] -> [...]
            for index in range(len(state)):
                if state[index] != 0:  # illegal moves are set to 0 desirability
                    prediction[index] = 0
            total = np.sum(prediction)
            prediction *= 1/abs(total)  # re-normalize after non-legal action indexes are set to 0

            action_index = np.where(prediction == np.max(prediction[np.nonzero(prediction)]))[0][0]  # Return index of maximum value in prediction excluding 0 that represents illegal action
        return self.get_action_from_index(action_index, state, player)

    @staticmethod
    def get_action_from_index(action_index, state, player):
        """ The target policy will return the index of the action with the highest probability,
         and this method uses the state and the player to produce the corresponding action = [cell_location, player]"""
        cell_row = floor(action_index/sqrt(len(state)))
        cell_col = int(action_index % sqrt(len(state)))
        chosen_action = [(cell_row, cell_col), player]
        return chosen_action

    def train(self, minibatch):
        """ Trains the  neural network on a given random minibatch from replay buffer. This minibatch contains several
         cases, where each case=(s,D) (i.e. a state and its distribution produced by MCTS)"""
        for i in range(len(minibatch)):
            x_train = convert_to_tensor(minibatch[i][0])
            y_train = convert_to_tensor(minibatch[i][1])
            self.anet.train(x_train, y_train)

    def decay_epsilon(self):
        """ Decay epsilon to reduce the amount of exploring in later episodes"""
        self.epsilon *= self.decay_rate

    def set_epsilon(self, value):
        """ Method used to set epsilon = 0 in the last episode to ensure exploitation"""
        self.epsilon = value

    def save(self, episode_num, dir_num):
        """Save current parameters (i.e. weights) of ANET for later use in tournament play"""
        if dir_num == 0:
            save_directory = "./models"  # demo models are saved in models directory
        else:
            save_directory = "./models/" + str(dir_num) + "_models"  # training models are saved in models_x directory
        path = save_directory + "/ANET_" + str(self.size) + "_ep_" + str(episode_num) + ".h5"  # h5 file format is recommended for storing NN parameters
        self.anet.model.save_weights(filepath=path)

    def load(self, path):
        """Load saved parameters of ANET for use in tournament play"""
        self.anet.model.load_weights(filepath=path)

"""def predict(self, state):  # TODO: Can this method be removed? Tournament uses target_policy
         Predicts the probability distribution of actions when given a state.
         This is used during the tournament to decide which actions should be chosen
        tensor_state = convert_to_tensor(state)
        return self.anet.predict(tensor_state)"""


class ANET:
    """ Class for making and training the neural network so it can be used to predict the action desirability
    for a given state. The choice of activation, optimizer and loss-function is customizable and is defined in configs"""
    def __init__(self, board_size, nn_dims, activation, optimizer, loss_function, learning_rate):
        self.input_size = 1 + board_size ** 2  # An indicator of the player is added to target value, so the size is equal board_size^2 + 1
        self.output_size = board_size ** 2
        self.hidden_layers_dim = nn_dims
        self.alpha = learning_rate
        self.activation = activation  # List of strings
        self.optimizer = self.get_optimizer(optimizer)
        self.loss_function = self.get_loss_function(loss_function)
        self.model = self.init_nn()

    def init_nn(self):
        """ Initializes the neural sequential model by adding layers and compiling the model."""
        model = KER.models.Sequential()
        model.add(KER.layers.Dense(self.input_size, input_shape=(self.input_size, )))
        for i in range(len(self.hidden_layers_dim)):
            model.add(KER.layers.Dense(self.hidden_layers_dim[i], activation=self.activation[i]))
        model.add(KER.layers.Dense(self.output_size, activation="softmax"))
        model.compile(optimizer=self.optimizer, loss=self.loss_function)
        # model.summary()
        return model

    def predict(self, tensor_state):
        """ Forward pass in neural model to produce the probability distribution over the given state"""
        return self.model(tensor_state)

    def train(self, x_train, y_train):  # backward-pass in neural network
        """ Backward pass in neural model to train the network when given a case from the Replay buffer. x_train is the
        state, while y_train is the target value (i.e. distribution produced by using node counters found in MCTS)"""
        self.model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=False, callbacks=[])


    """@staticmethod  # Not needed, since activation is only a string
    def get_activation_function(activation):
        if activation == "linear":
            return "linear"
        elif activation == "sigmoid":
            return "sigmoid"
        elif activation == "tanh":
            return "tanh"
        elif activation == "relu":
            return "relu"
            """

    def get_optimizer(self, optimizer):
        """ Allows for customizable optimizer defined in config"""
        if optimizer == "adam":
            return Adam(self.alpha)
        elif optimizer == "adagrad":
            return Adagrad(self.alpha)
        elif optimizer == "adadelta":
            return Adadelta(self.alpha)
        elif optimizer == "sgd":
            return SGD(self.alpha)
        elif optimizer == "RMSprop":
            return RMSprop(self.alpha)

    @staticmethod
    def get_loss_function(loss):
        """ Allows for customizable loss-function defined in config"""
        if loss == "mean-squared-error":
            return MeanSquaredError()
        elif loss == "KLDivergence":
            return KLDivergence()


def convert_to_tensor(state):
    """Convert given state from string format to tensor (i.e. array-like object) needed to be able to
    give the board state as input to the neural network"""
    state_array = np.array(list(state))  # make numpy array of string
    tensor_state = tf.convert_to_tensor(state_array, np.float32)  # convert numpy array to tensor
    return tf.convert_to_tensor(np.expand_dims(tensor_state, axis=0))  # insert axis on pos 0 to get a tensor shape that corresponds to the input_shape of the neural network (e.g. (15, ) --> (1, 15))
