import config
from environment.state_manager import StateManager
from agent.actor import Actor
from game_agent import GameAgent
from visualization import Visualizer
from agent.replay_buffer import ReplayBuffer


if __name__ == '__main__':
    state_manager = StateManager(config.game_type, config.board_size, nim_k=1)
    actor = Actor(config.learning_rate, config.epsilon, config.decay_rate, config.board_size, config.nn_dims, config.activation, config.optimizer, config.loss_function)
    visualizer = Visualizer(state_manager.game.get_board(), state_manager.game, config.visualization_speed, config.visualization_interval)
    replay_buffer = ReplayBuffer()
    game_agent = GameAgent(actor, config.save_interval, state_manager, visualizer, replay_buffer, config.starting_player, config.num_episodes, config.num_simulations)
    game_agent.run()



