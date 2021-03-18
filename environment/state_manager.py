from environment.hex_game import Hex
from environment.nim_game import Nim
from mcts_simulator.node import Node


class StateManager:
    def __init__(self, game_type, board_size, nim_k=1):
        self.size = board_size
        self.game_type = game_type
        self.game = self.init_game()
        self.nim_k = nim_k  # only used by Nim game, not by Hex

    def init_game(self):
        if self.game_type == "HEX":
            return Hex(self.size)
        elif self.game_type == "NIM":
            return Nim(self.size, self.nim_k)

    def get_child_nodes(self, state, player):
        """Generates child nodes when given the current state. The state of these nodes are decided by looking at
        the available legal actions (i.e. cells not owned by players) and these states are then given to the MCTS Node
        class to create node instances that can be used by the MCTS algorithm"""
        legal_actions = self.get_legal_actions(state, player)
        child_nodes = []
        for action in legal_actions:
            print(legal_actions)
            child_state = self.game.get_next_state(action)
            child_nodes.append(Node(child_state, action, player))

    def is_winning_state(self):
        """Recognize if game being played is in a winning state"""
        return self.game.is_winning_state()

    def check_winning_state(self, state, action):
        """Check if an action performed in a given state will lead to a winning state by setting
        the cells to the given state and then perform the action to produce the state that is checked.
        OBS: both cell states and actions are 2-tuples, so be careful to not confuse these"""
        self.game.set_cell_states(state)
        self.game.get_next_state(action)
        return self.is_winning_state()

    def perform_action(self, action):
        """Perform given action, where action = [cell_location, player]"""
        cell_location, player = action[0], action[1]
        self.game.perform_action(cell_location, player)

    def get_legal_actions(self, state, player):
        """Returns the available actions for the given player when the cells are in the given.
        The player is needed to return action = [cell_location, player]that can be used by MCTS.
        OBS: both cell states and actions are 2-tuples, so be careful to not confuse these"""
        legal_actions = []
        legal_cell_locations = self.game.get_legal_cell_locations(state)
        for cell in legal_cell_locations:
            action = [cell.get_location(), player]
            legal_actions.append(action)
        return legal_actions


if __name__ == '__main__':
    sm = StateManager("HEX", 3)
    sm.perform_action([(0, 2), 1])
    sm.perform_action([(0, 0), 2])
    sm.perform_action([(1, 1), 1])
    print(sm.get_child_nodes([(0, 1), (1, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], 1))

    sm.perform_action([(1, 0), 2])
    sm.perform_action([(1, 2), 1])
    #print(sm.game.get_current_state([(0, 1), (0, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)]))
    #print(sm.check_winning_state([(0, 1), (0, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], [(2, 0), 2]))
    print(sm.is_winning_state())
