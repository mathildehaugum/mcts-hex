from config import train_config, demo_config, oht_config, topp_config
from environment.state_manager import StateManager
from agent.actor import Actor
from game_agent import GameAgent
from visualization import Visualizer
from agent.replay_buffer import ReplayBuffer
from tournament import Topp


if __name__ == '__main__':

    run = "topp"  # "train" (build good agents), "demo" (4: run short training session), "topp" (5: tournament demo), "oht" (training of oht agent)

    if run == "train":
        for i in range(1, 5):
            state_manager = StateManager(train_config.game_type, train_config.board_size, nim_k=1)
            actor = Actor(train_config.learning_rate, train_config.epsilon, train_config.decay_rate, train_config.board_size, train_config.nn_dims, train_config.activation, train_config.optimizer, train_config.loss_function)
            visualizer = Visualizer(train_config.board_size, train_config.visualization_speed, train_config.visualization_interval)
            replay_buffer = ReplayBuffer()
            game_agent = GameAgent(actor, train_config.save_interval, state_manager, visualizer, replay_buffer, train_config.starting_player, train_config.num_episodes, train_config.num_simulations, i)  # i ≠ 0 to save training models in models_x
            game_agent.run()

    if run == "demo":
        state_manager = StateManager(demo_config.game_type, demo_config.board_size, nim_k=1)
        actor = Actor(demo_config.learning_rate, demo_config.epsilon, demo_config.decay_rate, demo_config.board_size, demo_config.nn_dims, demo_config.activation, demo_config.optimizer, demo_config.loss_function)
        visualizer = Visualizer(demo_config.board_size, demo_config.visualization_speed, demo_config.visualization_interval)
        replay_buffer = ReplayBuffer()
        game_agent = GameAgent(actor, demo_config.save_interval, state_manager, visualizer, replay_buffer, demo_config.starting_player, demo_config.num_episodes, demo_config.num_simulations, 0)  # i = 0 to save in demo models in models
        game_agent.run()

        topp = Topp("demo")
        topp.start_tournament()

    if run == "topp":
        topp = Topp("topp")
        topp.start_tournament()


