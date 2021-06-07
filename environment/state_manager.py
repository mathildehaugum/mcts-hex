from environment.hex_game import Hex
from environment.nim_game import Nim
from agent.node import Node
import numpy as np
from math import floor, sqrt


class StateManager:
    def __init__(self, game_type, board_size, nim_k=1):
        """ Class for making an interface between the Hex game logic and the MCTS/actor. The state manager will produce
        initial game states, generate child states from a parent state, recognize winning states, find legal actions
        for a given state and select most desirable actions for a given distribution."""
        self.size = board_size
        self.game_type = game_type
        self.game = None
        self.init_game()
        self.nim_k = nim_k  # only used by Nim game, not by Hex

    def init_game(self):
        """ Initializes a new game of the given game type"""
        if self.game_type == "HEX":
            self.game = Hex(self.size)
        elif self.game_type == "NIM":
            self.game = Nim(self.size, self.nim_k)

    def get_state(self):
        """ Returns the state of the cells on the current game board"""
        return self.game.get_current_state()

    def get_child_nodes(self, state, player):
        """ Generates child nodes of the given state. Each created child node contains a child state produced by
        performing one of the legal actions from the given state (i.e. fill cell with state = 0) and the legal
        action that created this child state"""
        child_nodes = []
        legal_actions = self.get_legal_actions(state, player)
        for action in legal_actions:
            child_state = self.game.get_next_state(state, action)
            child_nodes.append(Node(child_state, action))  # Player is included in action
        return child_nodes

    def get_legal_actions(self, state, player):
        """ Returns the available actions for the given player when the board is in the given state.
        The player is needed to return action = [cell_location, player] that can be used to create child nodes"""
        legal_actions = []
        legal_cells = self.game.get_legal_cells(state)  # cells with cell_state = 0
        for cell in legal_cells:
            action = [cell.get_location(), player]  # Action = [cell_location, player]
            legal_actions.append(action)
        return legal_actions

    def is_game_over(self, state):
        """Check if given state is a winning state by setting cells to the given state and running
        the winning state check of the game (e.g. dfs search in Hex)"""
        self.game.set_cell_states(state)
        return self.game.is_winning_state()

    def reset_state(self, state):
        """Used to reset cell states to state in root node"""
        self.game.set_cell_states(state)

    def get_next_state(self, state, action):
        """ Returns the state that is produced by performing the given action in the given state"""
        return self.game.get_next_state(state, action)

    def perform_action(self, action):
        """When given an action = [cell_location, player], this method will perform the action by
         using the method implemented by the game type (e.g. hex_game)"""
        self.game.perform_action(action)

    def select_action(self, root, player, normalized_counters):
        """Method for selecting the action that should be performed in the actual game. This will be the action with
        the highest normalized counter for p1 and lowest normalized counter for p2. The distribution includes 0 for
        illegal actions, so some array logic is used to locate the corresponding board action given the chosen index"""
        state_len = self.size ** 2
        if player == 1:
            action_index = np.where(normalized_counters == np.max(normalized_counters[np.nonzero(normalized_counters)]))[0][0]  # Returns index of action with highest counter excluding 0
        else:
            action_index = np.where(normalized_counters == np.min(normalized_counters[np.nonzero(normalized_counters)]))[0][0]  # Returns index of action with lowest counter excluding 0
        row = floor(action_index/sqrt(state_len))
        col = int(action_index % sqrt(state_len))
        for child in root.get_child_nodes():  # Finds the child node that contains the action = [(row, col), player] corresponding to the chosen (row, col) found above
            action_loc = child.get_action()[0]
            if (row, col) == action_loc:
                return child


def change_player(player):
    """ Switch the player"""
    return 1 if player == 2 else 2
