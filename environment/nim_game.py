from environment.game_board import HexagonalDiamondGrid
from visualization import *


class Nim:
    """ Class for performing a version of Nim where two players takes turn in adding a number of pieces
     between 1 and K. The game is over when only one empty cell remains and the current player is the loser.
     This version matches better with Hex compared to ordinary Nim where pieces are removed"""
    def __init__(self, board_size, nim_k=1):
        self.size = board_size
        self.num_pieces = self.size * self.size
        self.max_piece_adding = nim_k

        self.board = HexagonalDiamondGrid(self.size)

    def get_board(self):
        return self.board

    def perform_action(self, cell_locations, player):
        for i in range(len(cell_locations)):
            action = [cell_locations[i], player]
            self.board.perform_action(action)

    def is_winning_state(self):
        return self.board.get_pegs_nums[0] == 1

    def get_legal_actions(self):
        self.board.get_legac_actions()


if __name__ == '__main__':
    nim_game = Nim(4, 3)
    visualizer = Visualizer(nim_game.get_board(), 1000, nim_game)

    actions = [([(0, 0), (1, 0), (1, 1)], 1), ([(2, 1), (2, 2), (3, 3)], 2)]
    visualizer.visualize_last_game(actions)






# To match better with hex, look at version where players add a max number of pieces and when
# only one empty cell remains, current player is loser.


# red_pegs = pegs that player two removes
# blue_pegs = pegs that player one removes
# "holes" = black pegs left at board that still can be removed
# GAME OVER = just one white_peg left. Current player at this time is loser
# action = location of peg(s) to be removed, and player (decide if it gets red or blue). E.g ([(0, 1), (0, 0)], 1)
# action_list = list of action (i.e. nested list)