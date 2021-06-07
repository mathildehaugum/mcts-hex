from agent.mcts import MonteCarloTreeSearch
import random


class GameAgent:
    def __init__(self, actor, save_interval, state_manager, visualizer, replay_buffer, starting_player, num_episodes, num_simulations, dir_num=0):
        """ The game agent that performs the entire MCTS Algorithm on Hex games to train neural network models
        that can be used in later more intelligent plays. It also makes visualizations showing the chosen path of actions"""
        self.actor = actor  # 3: ANET with randomly initialized parameters
        self.save_interval = num_episodes/(save_interval-1)  # -1 since the first episode (i.e. num 0) is stored
        self.state_manager = state_manager
        self.visualizer = visualizer
        self.replay_buffer = replay_buffer
        self.starting_player = starting_player
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.dir_num = dir_num

    def run(self):
        """ Runs the entire algorithm connecting the state_manager, MCTS and actor to train the ANET that can be used in later plays"""

        self.actor.load("./models/ANET_6_ep_300.h5")

        # 2: Clear replay buffer
        self.replay_buffer.clear_buffer()

        # 4: For each episode (i.e. number_actual_games)
        for current_episode in range(1, self.num_episodes + 1):

            actions = []  # Used in visualization

            # 4a: Initialize game board to empty board
            self.state_manager.init_game()

            # 4b: Initialize starting_state and set root
            state = self.state_manager.get_state()

            # 4c: Initialize MCTS to root
            if self.starting_player == 0:
                player = random.choice([1, 2])
            else:
                player = self.starting_player
            mcts = MonteCarloTreeSearch(self.actor, self.state_manager, state, player)

            # 4D: While episode is not in final state (i.e. no player hos won)
            while not self.state_manager.is_game_over(state):
                for simulation in range(1, self.num_simulations + 1):
                    leaf_node = mcts.tree_search()
                    mcts.leaf_node_expansion(leaf_node)
                    final_evaluation = mcts.leaf_evaluation(leaf_node)
                    mcts.backpropagation(leaf_node, final_evaluation)

                D = mcts.get_root_distribution(mcts.get_root())

                self.replay_buffer.add_case(mcts.get_root(), D)

                self.state_manager.reset_state(state)  # Avoid that any cell states are changed during leaf_node evaluation (i.e. that search actions actually are performed
                chosen_child = self.state_manager.select_action(mcts.get_root(), player, D)  # select action to be executed in actual game
                actions.append(chosen_child.get_action())  # used in visualization

                self.state_manager.perform_action(chosen_child.get_action())
                state = self.state_manager.get_state()
                player = 1 if player == 2 else 2
                mcts.set_root(chosen_child)

            # 4e: Train ANET on random mini batch from buffer
            self.actor.train(self.replay_buffer.get_random_minibatch())
            self.actor.decay_epsilon()

            if current_episode == self.num_episodes:
                self.actor.set_epsilon(0)

            # 4f: Save ANET if condition is fulfilled
            if current_episode == 1 or current_episode % self.save_interval == 0:
                self.actor.save(current_episode, self.dir_num)

            # Visualize first episode and episodes within the interval defined in config
            if current_episode == 1 or current_episode % self.visualizer.get_interval() == 0:
                self.visualizer.visualize(actions, current_episode)
            print("Episode : " + str(current_episode) + ", epsilon: " + str(self.actor.epsilon))

















