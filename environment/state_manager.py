from environment.hex_game import Hex
from environment.nim_game import Nim
from agent.node import Node
import numpy as np
from math import floor, sqrt


class StateManager:
    def __init__(self, game_type, board_size, nim_k=1):
        self.size = board_size
        self.game_type = game_type
        self.game = None
        self.init_game()
        self.nim_k = nim_k  # only used by Nim game, not by Hex

    def init_game(self):
        """Initialize a new game of the given game type"""
        if self.game_type == "HEX":
            self.game = Hex(self.size)
        elif self.game_type == "NIM":
            self.game = Nim(self.size, self.nim_k)

    def get_state(self):
        """Returns the state of the cells on the current game board"""
        return self.game.get_current_state()

    def get_child_nodes(self, state, player):
        """Generates child nodes when given the current state. The state of these nodes are decided by looking at
        the available legal actions (i.e. cells not owned by players) and these states are then given to the MCTS Node
        class to create node instances that can be used by the MCTS algorithm"""
        child_nodes = []
        legal_actions = self.get_legal_actions(state, player)
        for action in legal_actions:
            child_state = self.game.get_next_state(state, action)
            child_nodes.append(Node(child_state, action))  # Player is included in action
        return child_nodes

    def get_legal_actions(self, state, player):
        """Returns the available actions for the given player when the cells are in the given state.
        The player is needed to return action = [cell_location, player]that can be used by MCTS."""
        legal_actions = []
        legal_cells = self.game.get_legal_cells(state)
        for cell in legal_cells:
            action = [cell.get_location(), player]  # Action = [cell_location, player]
            legal_actions.append(action)
        return legal_actions

    def is_winning_state(self, state):
        """Check if given state is a winning state by setting cells to the given state and running
        regular is_winning check."""
        self.game.set_cell_states(state)
        return self.game.is_winning_state()

    def reset_state(self, state):
        """Used to reset cell states to state in root node"""
        self.game.set_cell_states(state)

    def get_next_state(self, state, action):
        return self.game.get_next_state(state, action)

    def perform_action(self, action):
        """When given an action = [cell_location, player], this method will perform the action by
         using the method implemented by the game type (e.g. hex_game)."""
        self.game.perform_action(action)

    def select_action(self, root, player, normalized_counters):
        """Method for actually selecting the action that should be performed in the game. This
        will be the action with the highest counter for p1 and lowest counter for p2 """
        state_len = self.size ** 2
        print_value = ""
        if player == 1:
            action_index = np.where(normalized_counters == np.max(normalized_counters[np.nonzero(normalized_counters)]))[0][0]  # Returns index of action with highest counter excluding 0
            print_value = "max"
        else:
            action_index = np.where(normalized_counters == np.min(normalized_counters[np.nonzero(normalized_counters)]))[0][0]  # Returns index of action with lowest counter excluding 0
            print_value = "min"
        row = floor(action_index/sqrt(state_len))
        col = int(action_index % sqrt(state_len))
        for child in root.get_child_nodes():
            action_loc = child.get_action()[0]  # Chooses the child with action that produces the most (p1) or least (p2) desirable state
            if (row, col) == action_loc:
                #print("Node: " + root.name + ", Distribution: " + str(normalized_counters) + ", " + print_value + ": " + str(action_index) + " chosen action: " + str(child.get_action()) + '\n')
g                return child


def change_player(player):
    return 1 if player == 2 else 2
