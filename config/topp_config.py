# ----------------------------- BOARD PARAMETERS -----------------------------
game_type = "HEX"
board_size = 6
starting_player = 1  # Random starting player for each episode: 0, else 1/2


# ----------------------------- AGENT PARAMETERS -----------------------------
save_interval = 4  # M: cache interval in preparation for a TOPP
num_episodes = 210  # ensure this is divisible by save-interval - 1 (e.g. 200/4 = 50 => saved episodes are 0, 50, 100, 150, 200)
num_simulations = 800
exploration_c = 1


# ----------------------------- NN PARAMETERS -----------------------------
epsilon = 1
decay_rate = 0.973  # >= 0.95 to allow adequate exploration
learning_rate = 0.001
nn_dims = [64]
activation = ["relu"]  # softmax is recommended in requirements p. 10
optimizer = "adam"
loss_function = "crossentropy"


# ----------------------------- VISUALIZER PARAMETERS -----------------------------
verbose = False
visualize = True
visualization_speed = 1000
visualization_interval = 250
topp_is_visualized = True


# ----------------------------- TOPP PARAMETERS -----------------------------
load_path = "./models"
num_games = 100  # G: numbers of games to be played between agents in round-robin play of the TOPP
