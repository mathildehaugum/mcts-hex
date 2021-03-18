from environment.cell import Cell


class HexagonalDiamondGrid:
    """ Class for making a peg solitaire board which is a Hexagonal grid """
    def __init__(self, size):
        self.boardSize = size
        self.board = [[None for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.make_board()

    def make_board(self, ):
        """ Fills board with Cell objects and creates neighborhood following Diamond structure requirements"""
        for r in range(self.boardSize):
            for c in range(self.boardSize):  # avoid redundant calculation by adding neighbors "behind" current cell
                new_cell = Cell(r, c)
                self.board[r][c] = new_cell
                if c > 0:  # add left neighbor-cell
                    new_cell.add_neighbor(self.board[r][c-1])
                if r > 0:  # add above neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c])
                if r > 0 and c < self.boardSize-1:  # add right diagonal neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c+1])

    def reset_board(self):
        """ Reset board state by removing all holes """
        cell_list = self.get_cells()
        for current_cell in cell_list:
            current_cell.set_cell_state((0, 0))  # remove player ownership of cell

    def get_cell(self, location):
        """ Returns Cell object at given location if it exists """
        if 0 <= location[0] < self.boardSize and 0 <= location[1] < self.boardSize:
            return self.board[location[0]][location[1]]

    def get_cells(self):
        """ Returns all cells that are not None """
        cell_list = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None:
                    cell_list.append(current_cell)
        return cell_list

    def get_pegs(self):
        """ Returns three lists, one with cells that are empty (i.e. legal actions), one with cells that
         belong to player 1 and one with cells that belong to player 2"""
        empty_pegs = []
        red_pegs = []
        blue_pegs = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None:
                    if current_cell.get_cell_state() == (0, 0):
                        empty_pegs.append(current_cell)
                    elif current_cell.get_cell_state() == (1, 0):
                        red_pegs.append(current_cell)
                    elif current_cell.get_cell_state() == (0, 1):
                        blue_pegs.append(current_cell)
        return empty_pegs, red_pegs, blue_pegs

    def get_empty_cells(self):
        empty_cells = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None:
                    if current_cell.get_cell_state() == (0, 0):
                        empty_cells.append(current_cell)
        return empty_cells

    def get_pegs_nums(self):
        """ Returns number of empty pegs, red pegs and blue pegs"""
        empty_pegs, red_pegs, blue_pegs = self.get_pegs()
        return len(empty_pegs), len(red_pegs), len(blue_pegs)

    def get_binary_state(self):
        """ Returns space efficient and readable binary version of state where empty: 0, player 1: 1 and player 2: 2.
         OBS: Change to (0, 0), (1, 0) and (0, 1) later!"""
        binary_board_state = ""
        for current_cell in self.get_cells():
            if current_cell.get_cell_state() == (0, 0):
                binary_board_state += str((0, 0))
            if current_cell.get_cell_state() == (1, 0):
                binary_board_state += str((1, 0))
            if current_cell.get_cell_state() == (0, 1):
                binary_board_state += str((0, 1))
        return binary_board_state

    def perform_action(self, cell_location, player):
        """Given an action = [cell_location, player], performs action by changing state of given cell if it is not
        already owned by another player. Player moves that effect several cells are implemented by looping this method"""
        if self.get_cell(cell_location).get_cell_state() == (0, 0) and player == 1:
            self.get_cell(cell_location).set_cell_state((1, 0))
        elif self.get_cell(cell_location).get_cell_state() == (0, 0) and player == 2:
            self.get_cell(cell_location).set_cell_state((0, 1))


if __name__ == '__main__':
    board = HexagonalDiamondGrid(4)
    print(board.get_binary_state())
    board.perform_action([(0, 0), 1])
    board.perform_action([(0, 1), 2])
    board.perform_action([(2, 1), 1])
    board.perform_action([(2, 2), 2])
    print(board.get_cells())
    print(board.get_binary_state())





