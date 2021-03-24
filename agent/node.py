class Node:
    def __init__(self, state, action):
        """Initialize a node with its state and the given player that performs the given
         action from the parent node to produce this node"""
        self.name = "State: [" + str(state) + "], Action: " + str(action)
        self.state = state  # State when this node is root
        self.action = action  # Action taken from root leading to this child node
        self.parent = None
        self.child_nodes = []
        self.node_value = 0  # the value of the action that leads to this node
        self.node_counter = 0  # counts the number of times the node is visited during the search
        self.player = action[1]  # The player that perform the action that produces this node

    def __str__(self):
        """ Changes Node object representation to string format to make debugging easier"""
        return self.name

    def __repr__(self):
        """ Changes Node object representation to string format to make debugging easier"""
        return self.name

    def set_parent(self, parent_node):
        self.parent = parent_node

    def get_parent(self):
        return self.parent

    def add_child(self, child_node):
        self.child_nodes.append(child_node)
        child_node.set_parent(self)

    def get_child_nodes(self):
        return self.child_nodes

    def get_state(self):
        """Returns the state of the board game when this node is the root node"""
        return self.state

    def get_player(self):
        return self.player

    def set_player(self, player):
        self.player = player

    def get_action(self):
        return self.action

    def update_counter(self):
        self.node_counter += 1

    def get_counter(self):
        return self.node_counter

    def get_parent_counter(self):
        return self.parent.get_counter()

    def get_value(self):
        return self.node_value

    def update_value(self, value):
        self.node_value = value





