from environment.game_board import HexagonalDiamondGrid
import collections


class Hex:
    """ Class for performing a Hex game"""
    def __init__(self, board_size):
        self.size = board_size
        self.board = HexagonalDiamondGrid(self.size)

    def get_board(self):
        return self.board

    def get_current_state(self):
        return self.board.get_binary_state()

    def get_next_state(self, state, action):
        """Perform the given action to produce the next state and returns this state"""
        self.set_cell_states(state)
        self.perform_action(action)
        return self.board.get_binary_state()

    def get_legal_cells(self, state):
        """Given a state that lists the states of the cell on the board (e.g. [(0, 0), (1, 0), (0, 0), ...],
        this method will return the legal actions for this game state by updating the board cells and
        then returning the cells that are not owned by any player"""
        self.set_cell_states(state)
        return self.board.get_empty_cells()

    def set_cell_states(self, cell_states):
        """Cell_states is a list of the board cells state values, e.g. [(0, 1), (0, 0), (1, 0), (0, 0), ...]
         This method will receive such a list and changes the states of the currents board cells to
         the states given by the list"""
        board_cells = self.board.get_cells()
        # LEGG TIL EXCEPTION MANAGEMENT (sjekk at cell_states har like mange celler som board_cells)
        for i in range(len(board_cells)):
            board_cells[i].set_cell_state(cell_states[i])

    def perform_action(self, action):
        """Perform action by changing the state of the cell at the given location to a new state
        depending on the player: (1, 0): player 1, (0, 1): player 2"""
        cell_location, player = action[0], action[1]
        self.board.perform_action(cell_location, player)

    def is_winning_state(self):
        """The Hex game is in a winning state if one of the players has a complete path from one of their
        sides to the other. This is checked by iterating over the starting locations for each player sides and
        if one of these locations belong to the player that owns this side, a depth first search is performed to see
        if this piece is the beginning of a path to the other side.
        player 1 has left and right side, while player 2 has top and bottom"""
        for starting_location in range(self.size):
            p1_start_cell, p2_start_cell = self.board.get_cell([starting_location, 0]), self.board.get_cell([0, starting_location])
            if p1_start_cell.get_cell_state() == (1, 0):
                if self.depth_first_search(p1_start_cell):
                    return True, "p1"
            if p2_start_cell.get_cell_state() == (0, 1):
                if self.depth_first_search(p2_start_cell):
                    return True, "p2"
        return False, ""

    def depth_first_search(self, starting_cell):
        """Search after continuous path from given starting cell to a cell on the ending side
        for the player that owns the given cell"""
        cell_stack = collections.deque()  # implement python stack
        cell_stack.append(starting_cell)
        visited = set()  # set to keep track of visited cells
        while len(cell_stack) != 0:
            current_cell = cell_stack.pop()
            cell_state = current_cell.get_cell_state()
            cell_row, cell_col = current_cell.get_location()
            if (cell_state == (1, 0) and cell_col == self.size-1) or (cell_state == (0, 1) and cell_row == self.size-1):
                return True
            if current_cell not in visited:
                visited.add(current_cell)
                for neighbor in current_cell.get_neighbors():
                    if neighbor.get_cell_state() == starting_cell.get_cell_state():
                        cell_stack.append(neighbor)


"""if __name__ == '__main__':
    hex_game = Hex(3)
    #print(hex_game.get_board().get_binary_state())
    p1_action1 = [(1, 0), 1]
    p1_action2 = [(1, 1), 1]
    p1_action2 = [(1, 2), 1]
    p2_action1 = [(0, 0), 2]
    p2_action2 = [(0, 1), 2]
    p2_action2 = [(2, 1), 2]

    hex_game.perform_action((0, 2), 1)
    hex_game.perform_action((0, 0), 2)
    hex_game.perform_action((1, 1), 1)
    hex_game.perform_action((1, 0), 2)
    hex_game.perform_action((1, 2), 1)
    hex_game.perform_action((2, 0), 2)
    print(hex_game.get_legal_actions())
    #print(hex_game.get_board().get_binary_state())
    #print(hex_game.is_winning_state())"""






