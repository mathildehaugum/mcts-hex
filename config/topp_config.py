# ----------------------------- BOARD PARAMETERS -----------------------------
game_type = "HEX"
board_size = 4
starting_player = 1  # Random starting player for each episode: 0, else 1/2


# ----------------------------- AGENT PARAMETERS -----------------------------
save_interval = 4  # M: cache interval in preparation for a TOPP
num_episodes = 210  # ensure this is divisible by save-interval - 1 (e.g. 200/4 = 50 => saved episodes are 0, 50, 100, 150, 200)
num_simulations = 800
exploration_c = 1


# ----------------------------- NN PARAMETERS -----------------------------
epsilon = 1
decay_rate = 0.99  # >= 0.95 to allow adequate exploration
learning_rate = 0.005
nn_dims = [128, 128, 64]
activation = ["tanh", "tanh", "tanh"]  # softmax is recommended in requirements p. 10
optimizer = "adam"
loss_function = "mean-squared-error"


# ----------------------------- VISUALIZER PARAMETERS -----------------------------
verbose = False
visualize = True
visualization_speed = 1000
visualization_interval = 250


# ----------------------------- TOPP PARAMETERS -----------------------------
load_path = "./models"
num_games = 100  # G: numbers of games to be played between agents in round-robin play of the TOPP
