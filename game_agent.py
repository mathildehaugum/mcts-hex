from agent.actor import Actor
from agent.mcts import MonteCarloTreeSearch
from agent.replay_buffer import ReplayBuffer
from environment.state_manager import StateManager
from agent.node import Node


class GameAgent:
    def __init__(self, actor, save_interval, state_manager, visualizer, replay_buffer, starting_player, num_episodes, num_simulations):
        self.actor = actor  # 3: ANET with randomly initialized parameters
        self.save_interval = save_interval
        self.state_manager = state_manager
        self.visualizer = visualizer
        self.replay_buffer = replay_buffer
        self.starting_player = starting_player
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations

    def run(self):

        # 2: Clear replay buffer
        self.replay_buffer.clear_buffer()

        # 4: For each episode (i.e. number_actual_games)
        for current_episode in range(1, self.num_episodes + 1):

            actions = [] # Used in visualization

            # 4a: Initialize game board to empty board
            self.state_manager.init_game()

            # 4b: Initialize starting_state and set root
            state = self.state_manager.get_state()

            # 4c: Initialize MCT to root
            player = self.starting_player
            # CHANGE: MAKE NEW ROOT FROM STATE AND PLAYER AND GIVE THIS AS INPUT TO MCTS?
            mcts = MonteCarloTreeSearch(self.actor, self.state_manager, state, player)

            # 4D: While episode is not in final state (i.e. no player hos won)
            while not self.state_manager.is_winning_state(state):
                for simulation in range(1, self.num_simulations + 1):
                    leaf_node = mcts.tree_search()
                    mcts.leaf_node_expansion(leaf_node)  # OBSOBS: FJERNE RETURNERING? JA, GJORT!
                    final_evaluation = mcts.leaf_evaluation(leaf_node)
                    mcts.backpropagation(leaf_node, final_evaluation)

                D = mcts.get_root_distribution(mcts.get_root())

                self.replay_buffer.add_case(mcts.get_root(), D)

                self.state_manager.reset_state(state)  # Avoid that any cell states are changed during leaf_node evaluation (i.e. that search actions actually are performed
                chosen_child = mcts.select_action(player, D)
                actions.append(chosen_child.get_action())  # used in visualization

                self.state_manager.perform_action(chosen_child.get_action())
                state = self.state_manager.get_state()
                print(state)
                player = 1 if player == 2 else 2
                mcts.set_root(chosen_child)

            # 4e: Train ANET on random mini batch from buffer
            self.actor.train(self.replay_buffer.get_random_minibatch())
            self.actor.decay_epsilon()

            # 4f: Save ANET if condition is fulfilled
            if current_episode == 1 or current_episode % self.save_interval == 0:
                self.actor.save(current_episode)

            # Visualize last episode
            if current_episode % self.visualizer.get_interval():
                print(actions)
                self.visualizer.visualize(actions)
        self.actor.plot_loss_and_accuracy()


"""if __name__ == '__main__':
    actor = Actor()
    sm = StateManager("HEX", 3)
    buffer = ReplayBuffer()
    starting_player = 1
    num_episodes = 3
    num_simulations = 2

    agent = GameAgent(actor, 0, sm, None, buffer, starting_player, num_episodes, num_simulations)
    agent.run()"""
















