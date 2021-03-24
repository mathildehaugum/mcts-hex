from environment.hex_game import Hex
from environment.nim_game import Nim
from agent.node import Node
from math import floor
import numpy as np


class StateManager:
    def __init__(self, game_type, board_size, nim_k=1):
        self.size = board_size
        self.game_type = game_type
        self.game = self.init_game()
        self.nim_k = nim_k  # only used by Nim game, not by Hex

    def init_game(self):
        """Initialize a new game of the given game type"""
        if self.game_type == "HEX":
            return Hex(self.size)
        elif self.game_type == "NIM":
            return Nim(self.size, self.nim_k)

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
        The player is needed to return action = [cell_location, player]that can be used by MCTS.
        OBS: both cell states and actions are 2-tuples, so be careful to not confuse these"""
        legal_actions = []
        legal_cells = self.game.get_legal_cells(state)
        for cell in legal_cells:
            action = [cell.get_location(), player]  # Action = [cell_location, player]
            legal_actions.append(action)
        return legal_actions

    def is_winning(self):
        """Recognize if game being played is in a winning state"""
        return self.game.is_winning_state()

    def is_winning_state(self, state):
        """Check if given state is a winning state by setting cells to the given state and running
        regular is_winning check. OBS: both cell states and actions are 2-tuples, so be careful to not confuse these"""
        self.game.set_cell_states(state)
        return self.is_winning()

    def reset_state(self, state):
        """Used to reset cell states to state in root node"""
        self.game.set_cell_states(state)

    def perform_action(self, action):
        """When given an action index from the actor, this method will produce the action as [cell_location, player]
        and use the method implemented by the game type (e.g. hex_game) to perform this action."""
        self.game.perform_action(action)

    def perform_action_index(self, action_index, player):
        """When given an action index from the actor, this method will produce the action as [cell_location, player]
        and use the method implemented by the game type (e.g. hex_game) to perform this action."""
        action = self.index_to_action(action_index, player)
        self.game.perform_action(action)

    def get_next_state(self, state, action):
        return self.game.get_next_state(state, action)

    def get_next_state_action_index(self, state, action_index, player):
        """When given a state and an action index from the actor, this method will produce the action as
        [cell_location, player] and use the method implemented by the game type (e.g. hex_game) to get the next state
        produced by performing this action in the given state."""
        action = self.index_to_action(action_index, player)
        return self.game.get_next_state(state, action)

    def index_to_action(self, action_index, player):
        """The actor network outputs a number that corresponds to an index in a list of actions.
        This index needs to be translated into a cell_location that can be used to create a regular
        action = [cell_location, player]"""
        row = floor(action_index/self.size)
        col = action_index % self.size
        cell_location = [row, col]
        return [cell_location, player]

    """def use_heuristics(self, root_node, normalized_counters):
        state, children = root_node.get_state(), root_node.get_children()

        # In competition with external player, a winning strategy if mine player 1 begins is to play center
        if ((1, 0) not in state) or ((0, 1) not in state):
            normalized_counters = np.zeros(len(children))
            np.put(normalized_counters, len(children) // 2, 1.0)
            return normalized_counters

        # High normalized action counter can mean that the next state is the winning state.
        index = np.where(normalized_counters > 0.5)  # Returns an object that contains a list of indexes for elements that fulfill the condition
        if len(index[0]) > 0:  # if list of indexes that fulfill the condition is not empty
            best_action_index = index[0][0]
            child_action = children[best_action_index]
            child_state = self.game.get_next_state(state, child_action)
            if self.is_winning_state(child_state):
                normalized_counters = np.zeros(len(children))
                normalized_counters[best_action_index] = 1.0"""


"""if __name__ == '__main__':
    sm = StateManager("HEX", 3)
    sm.perform_action([(0, 2), 1])
    sm.perform_action([(0, 0), 2])
    sm.perform_action([(1, 1), 1])
    print(sm.get_child_nodes([(0, 1), (1, 0), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], 1))

    sm.perform_action([(1, 0), 2])
    sm.perform_action([(1, 2), 1])
    #print(sm.game.get_current_state([(0, 1), (0, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)]))
    #print(sm.check_winning_state([(0, 1), (0, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], [(2, 0), 2]))
    print(sm.is_winning_state())"""
