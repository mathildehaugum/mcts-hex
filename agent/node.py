class Node:
    def __init__(self, state, action):
        """ Class for making nodes that represents different states in the game. A node  will contain
        a state, the action that produced the state and the current player that will chose an action from this state"""
        self.name = "State: [" + str(state) + "], Action: " + str(action)
        self.state = state  # State when this node is root
        self.action = action  # Action taken from root leading to this child node
        self.parent = None
        self.child_nodes = []
        self.node_value = 0  # the value of the action that leads to this node reflecting the desirability of the state
        self.node_counter = 0  # counts the number of times the node is visited during MCTS
        if action[0] is None:  # starting player is given in the action
            self.player = action[1]
        else:
            self.player = 1 if action[1] == 2 else 2  # action[1] is the player that took the action that produced this state, while self.player is the player that performs an action from this node (i.e. decides next state)

    def __str__(self):
        """ Changes Node object representation to string format to make debugging easier"""
        return self.name

    def __repr__(self):
        """ Changes Node object representation to string format to make debugging easier"""
        return self.name

    def set_parent(self, parent_node):
        """ Sets the parent node containing the parent state"""
        self.parent = parent_node

    def get_parent(self):
        """ Returns the parent node containing the parent state"""
        return self.parent

    def add_child(self, child_node):
        """ Create parent-child relationships used during MCTS leaf expansion"""
        self.child_nodes.append(child_node)
        child_node.set_parent(self)

    def get_child_nodes(self):
        """ Returns the child nodes containing the available child states from the state of the current node"""
        return self.child_nodes

    def get_state(self):
        """ Returns the state of the board game when this node is the root node"""
        return self.state

    def get_player(self):
        """ Returns the player that will chose an action from the state of this node"""
        return self.player

    def set_player(self, player):
        """ Sets the player that will chose an action from the state of this node"""
        self.player = player

    def get_action(self):
        """ Return the action that produced this node"""
        return self.action

    def update_counter(self):
        """ Increase the counter of the node, representing how many times the state has been
        visited during MCTS. The counters are updated during backpropagation"""
        self.node_counter += 1

    def get_counter(self):
        """ Returns the counter of the node, used to produce the action probability distribution in MCTS"""
        return self.node_counter

    def get_parent_counter(self):
        """ Returns the counter of the parent node, used to decide the path during the tree search step of MCTS"""
        return self.parent.get_counter()

    def get_value(self):
        """ Returns the value of the node, used to decide the path during the tree search step of MCTS"""
        return self.node_value

    def update_value(self, value):
        """ Updates the value of the node"""
        self.node_value = value





