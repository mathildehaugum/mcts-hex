from environment.hex_game import Hex
from environment.nim_game import Nim
from agent.node import Node


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

    def get_next_state(self, state, action):
        return self.game.get_next_state(state, action)
