import math
from BasicClientActorAbs import BasicClientActorAbs
from agent.actor import Actor
from config import train_config


class BasicClientActor(BasicClientActorAbs):
    def __init__(self, actor, IP_address=None, verbose=True):
        self.series_id = -1
        self.actor = actor
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        player = state[0]
        board_state = state[1:]
        next_move = self.actor.target_policy(board_state, player, is_top_policy=True)
        return next_move[0]

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return
        """
        self.series_id = series_id
        print("Your ID: " + str(series_id))
        self.actor.load("../models/ANET_6_ep_225.h5")

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':

    # Actor parameters
    board_size = train_config.board_size
    learning_rate = train_config.learning_rate
    epsilon = train_config.epsilon
    decay_rate = train_config.decay_rate
    nn_dims = train_config.nn_dims
    activation = train_config.activation
    optimizer = train_config.optimizer
    loss_function = train_config.loss_function
    actor = Actor(learning_rate, epsilon, decay_rate, board_size, nn_dims, activation, optimizer, loss_function)

    bsa = BasicClientActor(actor, verbose=False)
    bsa.connect_to_server()
