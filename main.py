import config
from environment.state_manager import StateManager
from agent.actor import Actor
from game_agent import GameAgent
from visualization import Visualizer
from agent.replay_buffer import ReplayBuffer
from tournament import Topp


if __name__ == '__main__':
    #for i in range(1, 5):
        # Reinforcement learning:
        """state_manager = StateManager(config.game_type, config.board_size, nim_k=1)
        actor = Actor(config.learning_rate, config.epsilon, config.decay_rate, config.board_size, config.nn_dims, config.activation, config.optimizer, config.loss_function,  dir_num=i)
        visualizer = Visualizer(config.board_size, config.visualization_speed, config.visualization_interval)
        replay_buffer = ReplayBuffer()
        game_agent = GameAgent(actor, config.save_interval, state_manager, visualizer, replay_buffer, config.starting_player, config.num_episodes, config.num_simulations, i)
        game_agent.run()"""

        # Tournament of Progressive Policies (TOPP):
        topp = Topp()
        topp.start_tournament()
