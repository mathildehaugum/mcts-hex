from numpy import log, sqrt
from sklearn import preprocessing
import numpy as np
from agent.node import Node


class MonteCarloTreeSearch:
    def __init__(self, actor, state_manager, init_state, root_player, c=1):
        self.root = self.create_root(init_state, root_player)
        self.actor = actor  # Responsible for updating the target policy = default policy (on-policy) used during rollout
        self.c = c  # exploration constant used to find exploration bonus u(s,a)
        self.state_manager = state_manager
        self.state_manager.init_game()

    @staticmethod
    def create_root(state, root_player):
        return Node(state, (None, root_player))

    def get_root(self):
        return self.root

    def set_root(self, root):
        self.root = root

    def tree_search(self):
        """Traverse the tree from a root to a leaf node by using the tree policy. As long as the root does not produce
         a winning state (i.e. is not a leaf node) and it has one or more children, the method will update the values of
         the child nodes and then use the Tree policy (min-max) to choose the next root"""
        current_root = self.root
        child_nodes = current_root.get_child_nodes()
        while not self.state_manager.is_winning_state(current_root.get_state()) and len(child_nodes) > 0:
            for child in child_nodes:  # Update value of each child node before choosing the next root
                if current_root.get_player() == 1:
                    u = self.c * sqrt(log(child.get_parent_counter()/(1 + child.get_counter())))
                    child.update_value(child.get_value() + u)
                elif current_root.get_player() == 2:
                    u = self.c * sqrt(log(child.get_parent_counter()/(1 + child.get_counter())))
                    child.update_value(child.get_value() - u)

            # Tree policy: choose action (and hence next root) that maximize value for P1 or minimize value for P2
            if current_root.get_player() == 1:
                current_root = max(child_nodes, key=lambda item: item.get_value())
            elif current_root.get_player() == 2:
                current_root = min(child_nodes, key=lambda item: item.get_value())
            child_nodes = current_root.get_child_nodes()
        return current_root

    def leaf_node_expansion(self, node):
        """Some or all child states of a parent state is generated, and the tree node housing the parent state
        (i.e. parent node) is connected to the nodes housing the child states (i.e. child nodes)"""
        if not self.state_manager.is_winning_state(node.get_state()):  # node is a leaf node, so expansion is not possible
            child_nodes = self.state_manager.get_child_nodes(node.get_state(), node.get_player())
            child_player = 1 if node.player == 2 else 2
            for child in child_nodes:
                node.add_child(child)  # connect child to parent, and parent to child (both performed by add_child())
                child.set_player(child_player)

    def leaf_evaluation(self, leaf_node):
        """The value of a leaf node is estimated by performing a rollout simulation, using the target
        policy (i.e. default policy) from the leaf node to a final state (i.e. winning state)"""
        state = leaf_node.get_state()
        player = leaf_node.get_player()
        while not self.state_manager.is_winning_state(state):
            chosen_action = self.actor.target_policy(state, player)
            state = self.state_manager.get_next_state(state, chosen_action)
            player = 1 if player == 2 else 2
        if player == 2:  # In winner state, the previous player will be the winner
            reward = 1
        else:
            reward = 0  # Maybe use -1?
        return reward

    @staticmethod
    def backpropagation(node, final_evaluation):
        """The evaluation of a final state is propagated back up the tree, and the value and the counter of
        the nodes that are located on the path to the root are updated"""
        node.update_counter()
        while node.get_parent() is not None:
            node.get_parent().update_counter()  # N(s,a) = N(s,a) + 1
            node_eval = node.get_value()
            node_eval += final_evaluation  # E_t = E_t + eval (p. 7)
            node.update_value(node_eval/node.get_counter())  # Q(s,a) = E_t/N(s,a) (p. 7)
            node = node.get_parent()

    def select_action(self, player, normalized_counters):
        """Method for actually selecting the action that should be performed in the game. This
        will be the action with the highest counter for p1 and lowest counter for p2 """
        if player == 1:
            action_index = np.argmax(normalized_counters)
        else:
            action_index = np.argmin(normalized_counters)
        chosen_child = self.root.get_child_nodes()[action_index]
        return chosen_child

    def get_root_distribution(self, root_node):
        """Method for normalizing the action counters from the root (i.e. edges to child nodes) to produce
        a probability distribution that can be used as target for training the actor network"""
        action_counter = []
        for child in root_node.get_child_nodes():
            action_counter.append(child.get_counter())
        numpy_counters = np.asarray(action_counter, dtype=np.float32)
        normalized_counters = preprocessing.normalize([numpy_counters])
        # normalized_counters = self.state_manager.use_heuristics(root_node, normalized_counters)  # possibly apply heuristic to distribution
        return normalized_counters






