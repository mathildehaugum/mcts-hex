from numpy import log, sqrt
import numpy as np
from agent.node import Node
from math import floor, sqrt


class MonteCarloTreeSearch:
    """ Class for performing Monte Carlo Tree Search (MCTS) to identify the most desirable action in each step of an episode,
    which is achieved by performing four steps: 1) Tree search, 2) Node expansion, 3) Leaf evaluation and 4) Backpropagation.
    The aim of this search is to update the counters depending on how many times a state is visited during simulation,
    and these counters are used to produce target values for training of actor neural network."""
    def __init__(self, actor, state_manager, init_state, root_player, c=1):
        self.root = self.create_root(init_state, root_player)
        self.actor = actor  # Responsible for updating the target policy = default policy (on-policy) used during rollout
        self.c = c  # exploration constant used to find exploration bonus u(s,a)
        self.state_manager = state_manager

    @staticmethod
    def create_root(state, root_player):
        """ Method for creating the root node that represents the actual state of the game. MCTS is used to
        identify the most desirable action from this state"""
        return Node(state, (None, root_player))

    def get_root(self):
        """ Returns the root of the MCTS"""
        return self.root

    def set_root(self, root):
        """ Sets the root of the MCTS"""
        self.root = root

    def tree_search(self):
        """ Traverse the tree from a root to a leaf node by using the tree policy. As long as the root does not produce
        a winning state (i.e. is a final node) and it has childrens (i.e. is not a leaf node), the method will update the
        values of the child nodes and then use the Tree policy (min-max) to choose the next node. It returns the chosen leaf node"""
        current_root = self.root
        child_nodes = current_root.get_child_nodes()
        while not self.state_manager.is_winning_state(current_root.get_state()) and len(child_nodes) > 0:
            action_values = {}
            for child in child_nodes:  # Update value of each child node before using tree policy to choose the next root
                if current_root.get_player() == 1:
                    u = self.c * sqrt(log(child.get_parent_counter())/(1 + child.get_counter()))
                    action_values[child] = child.get_value() + u
                elif current_root.get_player() == 2:
                    u = self.c * sqrt(log(child.get_parent_counter())/(1 + child.get_counter()))
                    action_values[child] = child.get_value() - u
            # Tree policy: choose action (and hence next root) that maximize value for P1 or minimize value for P2
            if current_root.get_player() == 1:  # TODO: IF ACTION_VALUES CONTAINS SIMILAR VALUES, THE FIRST IS ALWAYS CHOSEN, IS THAT OKAY?
                current_root = max(action_values, key=action_values.get)  # P1 chooses action argmax(Q + u)
            elif current_root.get_player() == 2:
                current_root = min(action_values, key=action_values.get)  # P2 chooses action argmin(Q - u)
            child_nodes = current_root.get_child_nodes()
        return current_root  # The chosen leaf node

    def leaf_node_expansion(self, node):
        """ The child states of a parent state is generated, and the tree node housing the parent state
        (i.e. parent node) is connected to the nodes housing the child states (i.e. child nodes)"""
        if not self.state_manager.is_winning_state(node.get_state()):  # node is a leaf node, so expansion is not possible
            child_nodes = self.state_manager.get_child_nodes(node.get_state(), node.get_player())
            child_player = 1 if node.player == 2 else 2
            for child in child_nodes:
                node.add_child(child)  # connect child to parent, and parent to child (both performed by add_child())
                child.set_player(child_player)

    def leaf_evaluation(self, leaf_node):
        """ The value of a leaf node is estimated by performing a rollout simulation, using the target
        policy (i.e. default policy) from the leaf node to a final state (i.e. winning state)."""
        state = leaf_node.get_state()
        player = leaf_node.get_player()
        while not self.state_manager.is_winning_state(state):
            chosen_action = self.actor.target_policy(state, player)  # Default policy = target policy since MCTS is on-policy. This policy is created by the actor NN
            state = self.state_manager.get_next_state(state, chosen_action)
            player = 1 if player == 2 else 2
        if player == 2:  # In winner state, the previous player will be the winner
            reward = 1
        else:
            reward = -1  # Maybe use 0?
        return reward

    @staticmethod
    def backpropagation(node, final_evaluation):
        """ The evaluation of a final state is propagated back up the tree, and the value and the counter of
        the nodes that are located on the path to the root are updated. These counters are later used to produce
        the action probability distribution used as target in training of the actor NN (ANET)"""
        node.update_counter()
        while node.get_parent() is not None:
            node.get_parent().update_counter()  # N(s,a) = N(s,a) + 1
            node_eval = node.get_value()
            node_eval += final_evaluation  # E_t = E_t + eval (p. 7)
            node.update_value(node_eval/node.get_counter())  # Q(s,a) = E_t/N(s,a) (p. 7)
            node = node.get_parent()

    def get_root_distribution(self, node):
        """ Method for normalizing the action counters from the root (i.e. edges to child nodes) to produce
        a probability distribution that can be used as target for training the actor network (ANET)"""

        # OLD
        distribution = [0]*self.state_manager.size**2  # The value of illegal actions will be 0, since there is no child node for illegal actions so their value in the distribution is never changed from 0
        for child in node.get_child_nodes():
            action_location = child.action[0]  # The action that produces the child state from node state
            row, col = action_location
            distribution[row * self.state_manager.size + col] = child.get_counter()

        numpy_counters = np.asarray(distribution, dtype=np.float32)
        counter_sum = np.sum(numpy_counters)
        normalized_counters = numpy_counters/counter_sum
        return normalized_counters


"""        # NEW
        board_size = size
        distributions = [0]*board_size**2
        for i, child in enumerate(node.get_child_nodes()):
            col, row = child.action[0]
            distributions[row * board_size + col] = child.get_counter()
        total = np.sum(distributions)
        normalized_distribution = np.array(distributions)/total
        D = tuple(normalized_distribution)
        print("NEW: " + str(D))
        return tuple(D)
        
        
         action_counter.append(child.get_counter())
    # print("Root: " + str(root_node) + ", Action counters: " + str(action_counter))
    numpy_counters = np.asarray(action_counter, dtype=np.float32)
    counter_sum = np.sum(numpy_counters)
    normalized_counters = numpy_counters/counter_sum
"""







