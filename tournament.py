import glob
import os
import random
from config import topp_config, demo_config
from agent.actor import Actor
from environment.state_manager import StateManager
from visualization import Visualizer
ROOT_DIR = os.path.abspath(os.curdir)  # Needed to locate images directory to store board visualizations


class Topp:
    def __init__(self, config_type, topp_visualization=True):
        """ Class for executing a tournament of progressive policies (TOPP) with the aim of showing an improvement
        in the target policy. Improvement entails that policies with less training perform more poorly than policies
        with more training. There are however many impeding factors, so the progress can have some variations"""
        if config_type == "topp":
            config = topp_config
        else:
            config = demo_config
        self.state_manager = StateManager(config.game_type, config.board_size, nim_k=1)
        self.visualizer = Visualizer(config.board_size, config.visualization_speed, config.visualization_interval)
        self.topp_visualization = topp_visualization
        self.G = config.num_games
        self.load_directory = config.load_path
        self.agents = self.init_agents(config)
        self.results = self.init_results()

    def init_agents(self, config):
        """ Makes an actor for each file and load the actor with the weights saved in the corresponding file"""
        agents = []
        os.chdir(self.load_directory)  # Changes current working directory to load_directory
        file_list = glob.glob("*.h5")  # Returns all files in directory with ending .h5
        file_list = sorted(file_list, key=return_episode_num)
        for file in file_list:
            filename = "../models/" + file  # The path is used in actor which is placed in agent
            actor = Actor(config.learning_rate, config.epsilon, config.decay_rate, config.board_size, config.nn_dims, config.activation, config.optimizer, config.loss_function, filename)
            actor.load(filename)
            agents.append(actor)
        return agents

    def init_results(self):
        """ Initializes the score board that keeps track of how many games each actor has won in the entire tournament"""
        results = {}
        for i in range(len(self.agents)):
            actor = self.agents[i]
            actor.set_name("Actor")  # the filename part of the name is set in actor using filename from init_agents
            results[actor.name] = 0
        return results

    def start_tournament(self):
        """ Runs the tournament by getting each player to play a given number of games against each of the other players.
        The result, showing how many times each agent won, is printed at the end of the tournament"""
        for i in range(0, len(self.agents)):
            for j in range(i+1, len(self.agents)):
                p1, p2 = self.agents[i], self.agents[j]
                p1_total_win = 0
                p2_total_win = 0
                for game_num in range(self.G):
                    p1_wins, p2_wins, actions = self.play_game(p1, p2)
                    p1_total_win += p1_wins
                    p2_total_win += p2_wins
                print(p1.name + ": " + str(p1_total_win) + " wins, " + p2.name + ": " + str(p2_total_win) + " wins")
                if self.topp_visualization:
                    p1_num = p1.filename.split("ep_")[1].split(".h5")[0]
                    p2_num = p2.filename.split("ep_")[1].split(".h5")[0]
                    os.chdir(ROOT_DIR)
                    self.visualizer.visualize(actions, p1_num + "_" + p2_num)
        self.print_result()

    def play_game(self, p1, p2):
        """ Execute a game of Hex between two given players. Each player represents an actor loaded with a pretrained
        model, where the target policy of the actor is used to choose the most desirable action given the current state"""
        self.state_manager.init_game()
        state = self.state_manager.get_state()
        players = [p1, p2]
        player = random.choice([1, 2])
        actions = []
        p1_wins = 0
        p2_wins = 0
        while not self.state_manager.is_game_over(state):
            current_agent = players[player - 1]
            actor_chosen_action = current_agent.target_policy(state, player, is_top_policy=True)  # is_top_policy = True to ensure that the agents uses the ANET and not the random exploration
            actions.append(actor_chosen_action)
            self.state_manager.perform_action(actor_chosen_action)

            state = self.state_manager.get_state()
            player = change_player(player)
        if player == 1:
            p2_wins += 1
        else:
            p1_wins += 1
        winning_agent = players[change_player(player)-1]  # Since player is changed in end of while, the winning player at winning state is the previous player
        # print(p1.name + " vs. " + p2.name + ", winner: " + winning_agent.name + ", actions: " + str(actions))
        self.results[winning_agent.name] += 1
        return p1_wins, p2_wins, actions

    def print_result(self):
        """ Prints the result showing how many times each actor has won in the entire tournament"""
        print("Final results: ")
        for i in range(1, len(self.agents) + 1):
            agent = self.agents[i-1]
            print(agent.name + ": {} wins".format(self.results[agent.name]))


def return_episode_num(name):
    """ Method used to sort the file after increasing episode number"""
    return int(name.split(".")[0].split("ep_")[1])  # Use split to return only the episode number needed to sort the files in increasing order


def change_player(player):
    """ Swap players"""
    return 1 if player == 2 else 2
