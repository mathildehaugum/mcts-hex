# ----------------------------- BOARD PARAMETERS -----------------------------
game_type = "HEX"
board_size = 3
starting_player = 1


# ----------------------------- AGENT PARAMETERS -----------------------------
save_interval = 5
num_episodes = 10
num_simulations = 3
exploration_c = 1


# ----------------------------- NN PARAMETERS -----------------------------
epsilon = 1
decay_rate = 0.9
learning_rate = 0.01
nn_dims = [26, 512, 256, 25]
activation = "relu"
optimizer = "adam"
loss_function = "mean-squared-error"


# ----------------------------- VISUALIZER PARAMETERS -----------------------------
verbose = False
visualize = True
visualization_speed = 1000
visualization_interval = 5


