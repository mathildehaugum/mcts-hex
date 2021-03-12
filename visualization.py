import networkx as nx
import matplotlib.pyplot as plt
from celluloid import Camera


class Visualizer:
    def __init__(self, board, player, visualization_speed):
        self.board = board
        self.player = player
        self.camera = Camera(plt.figure())
        self.G, self.pos = self.init_board_visualizer()
        self.speed = visualization_speed

    def init_board_visualizer(self):
        G = nx.Graph()
        for current_cell in self.board.get_cells():
            x_pos = current_cell.get_location()[0]  # Positive x-pos to not horizontally flip graph
            y_pos = -current_cell.get_location()[1]  # Negative y-pos to vertically flip graph
            G.add_node(current_cell, pos=(x_pos, y_pos))
            for neighbor_cell in current_cell.get_neighbors():
                G.add_edge(current_cell, neighbor_cell)  # edges can be added before nodes (networkx docs)
        return G, nx.get_node_attributes(G, 'pos')

    def draw_board(self):
        """Method for drawing board state called after every player move that
        will change holes, red_pegs and black_pegs content"""
        holes, red_pegs, black_pegs = self.board.get_pegs()
        nx.draw_networkx_edges(self.G, self.pos)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[holes],
                               node_color='white', linewidths=2, edgecolors='black')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[red_pegs],
                               node_color='red', linewidths=2, edgecolors='black')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[black_pegs],
                               node_color='black', linewidths=2, edgecolors='black')
        self.campera.snap()

    def animate_visualiser(self):
        animation = self.camera.animate(interval=self.speed, repeat=False)
        animation.save('images/animation.gif')
        plt.show(block=False)
        plt.close()

    def visualize_episode(self, current_episode_saps):
        """ Visualizes the performed actions and resulting states in the given episode"""
        for state_action_pair in current_episode_saps:
            self.draw_state_transition(state_action_pair[1])
            self.player.perform_action(state_action_pair[1])
            self.draw_board()
        self.animate_visualiser()


