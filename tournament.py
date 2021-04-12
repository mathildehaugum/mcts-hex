import glob
import os
import random
import config
from agent.actor import Actor
from environment.state_manager import StateManager
from visualization import Visualizer


class Topp:
    def __init__(self):
        self.state_manager = StateManager(config.game_type, config.board_size, nim_k=1)
        self.visualizer = Visualizer(config.board_size, config.visualization_speed, config.visualization_interval)
        self.G = config.num_games
        self.load_directory = config.load_path
        self.agents = self.init_agents()
        self.results = self.init_results()

    def init_agents(self):
        agents = []
        os.chdir(self.load_directory)  # Changes current working directory to load_directory
        file_list = glob.glob("*.h5")  # Returns all files in directory with ending .h5
        file_list = sorted(file_list, key=return_episode_num)
        for file in file_list:
            filepath = "../models/" + file  # The path is used in actor which is placed in agent
            actor = Actor(config.learning_rate, config.epsilon, config.decay_rate, config.board_size, config.nn_dims, config.activation, config.optimizer, config.loss_function, filepath)
            actor.load(filepath)
            agents.append(actor)
        return agents

    def init_results(self):
        results = {}
        for i in range(len(self.agents)):
            actor = self.agents[i]
            actor.set_name("Actor")
            results[actor.name] = 0
        return results

    def start_tournament(self):
        for i in range(0, len(self.agents)):
            for j in range(i+1, len(self.agents)):
                p1, p2 = self.agents[i], self.agents[j]
                p1_total_win = 0
                p2_total_win = 0
                for game_num in range(self.G):
                    p1_wins, p2_wins = self.play_game(p1, p2)
                    p1_total_win += p1_wins
                    p2_total_win += p2_wins
                print(p1.name + ": " + str(p1_total_win) + " wins, " + p2.name + ": " + str(p2_total_win) + " wins")
        self.print_result()

    def play_game(self, p1, p2):
        self.state_manager.init_game()
        state = self.state_manager.get_state()
        players = [p1, p2]
        player = random.choice([1, 2])
        actions = []
        p1_wins = 0
        p2_wins = 0
        while not self.state_manager.is_winning_state(state):
            current_agent = players[player - 1]
            actor_chosen_action = current_agent.target_policy(state, player, is_top_policy=True)
            actions.append(actor_chosen_action)
            self.state_manager.perform_action(actor_chosen_action)

            state = self.state_manager.get_state()
            player = change_player(player)
        if player == 1:
            p2_wins += 1
        else:
            p1_wins += 1
        winning_agent = players[change_player(player)-1]  # Since player is changed in end of while, the winning player at winning state is the previous player
        #print(p1.name + " vs. " + p2.name + ", winner: " + winning_agent.name + ", actions: " + str(actions))
        self.results[winning_agent.name] += 1
        return p1_wins, p2_wins

    def print_result(self):
        print("Final results: ")
        for i in range(1, len(self.agents) + 1):
            agent = self.agents[i-1]
            print(agent.name + ": {} wins".format(self.results[agent.name]))


def return_episode_num(name):
    return int(name.split(".")[0].split("ep_")[1])  # Use split to return only the episode number needed to sort the files in increasing order


def change_player(player):
    return 1 if player == 2 else 2
