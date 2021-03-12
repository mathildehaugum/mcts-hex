from environment.game_board import *

class Nim:
    """ Class for performing a version of Nim where two players takes turn in adding a number of pieces
     between 1 and K. The game is over when only one empty cell remains and the current player is the loser.
     This version matches better with Hex (compared to ordinary Nim where pieces are removed)"""
    def __init__(self, board, nim_n, nim_k=1, ):
        self.num_pieces = nim_n
        self.max_piece_adding = nim_k
        self.board = board

    def perform_actions(self, cell_list, player):
        for i in range(len(cell_list)):
            action = [cell_list[i], player]
            print(action)
            self.board.perform_action(action)

    def is_winning_state(self):
        return self.board.get_pegs_nums[0] == 1


if __name__ == '__main__':
    board = HexagonalDiamondGrid(4)
    nim_game = Nim(board, 16, 3)
    print(board.get_binary_state())
    nim_game.perform_actions([(0, 0), (1, 0), (1, 1)], 1)
    print(board.get_binary_state())
    nim_game.perform_actions([(2, 1), (2, 2), (3, 3)], 2)
    print(board.get_binary_state())




# To match better with hex, look at version where players add a max number of pieces and when
# only one empty cell remains, current player is loser.


# red_pegs = pegs that player two removes
# blue_pegs = pegs that player one removes
# "holes" = black pegs left at board that still can be removed
# GAME OVER = just one white_peg left. Current player at this time is loser
# action = location of peg(s) to be removed, and player (decide if it gets red or blue). E.g ([(0, 1), (0, 0)], 1)
# action_list = list of action (i.e. nested list)