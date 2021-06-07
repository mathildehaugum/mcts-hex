import networkx as nx
import matplotlib.pyplot as plt
from celluloid import Camera
from environment.hex_game import Hex


class Visualizer:
    def __init__(self, board_size,  visualization_speed, visualization_interval):
        """ Class for visualizing a diamond hexagonal grid and the actions that are performed in the game"""
        self.game = Hex(board_size)
        self.board = self.game.get_board()
        self.camera = Camera(plt.figure())
        self.G, self.pos = self.init_board_visualizer()
        self.speed = visualization_speed
        self.interval = visualization_interval

    def get_interval(self):
        """ Returns the interval describing how often a game visualization should be made"""
        return self.interval

    def init_board_visualizer(self):
        """ Initializes the board visualization, where array logic is used to create a diamond structure with the
        correct rotation and direction"""
        G = nx.Graph()
        row = 0
        col = self.game.size - 1  # Since location placements start with 0, the size have to be adjusted
        for current_cell in self.board.get_cells():  # Cells are given in order of increasing col, and then increasing row [00, 01, 02, 10, 11, 12, 20, 21, 22]
            if row == self.game.size:
                row = 0
                col -= 1
            y_pos = -(current_cell.get_location()[0] + row)  # Positive x-pos to not horizontally flip graph
            x_pos = (current_cell.get_location()[1] + col)  # Negative y-pos to vertically flip graph
            G.add_node(current_cell, pos=(x_pos, y_pos))
            row += 1
            for neighbor_cell in current_cell.get_neighbors():
                G.add_edge(current_cell, neighbor_cell)  # edges can be added before nodes (networkx docs)
        return G, nx.get_node_attributes(G, 'pos')

    def draw_board(self):
        """Method for drawing board state called after every player move that
        will change holes, red_pegs and black_pegs content"""
        holes, red_pegs, black_pegs = self.board.get_pegs()
        nx.draw_networkx_edges(self.G, self.pos)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=holes,
                               node_color='white', linewidths=2, edgecolors='black')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=red_pegs,
                               node_color='red', linewidths=2, edgecolors='black')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=black_pegs,
                               node_color='black', linewidths=2, edgecolors='black')
        plt.ylabel("P1                                                P2")
        plt.title('Hex')

        self.camera.snap()

    def animate_visualiser(self, current_episode):
        """ Creates and saves a gif of the created images"""
        animation = self.camera.animate(interval=self.speed, repeat=False)
        animation.save('images/animation_' + str(current_episode) + '.gif')
        plt.show(block=False)
        plt.close()

    def visualize(self, game_actions, current_episode):
        """ Visualizes the performed actions and resulting states in the given episode"""
        self.board.reset_board()
        self.draw_board()
        for action in game_actions:
            self.game.perform_action(action)
            self.draw_board()
        self.animate_visualiser(current_episode)



