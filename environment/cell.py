class Cell:
    """ Class for making a Hex piece which is a Cell object"""
    def __init__(self, row, column):
        self.location = (row, column)
        self.name = "Cell" + str(row) + str(column)
        self.neighbor_list = []
        self.cell_state = 0  # tells the state of the cell, where 0 is empty, 1 belong to player 1 and 2 belong to player 2

    def __str__(self):
        """ Changes Cell object representation to string format to make debugging easier"""
        return self.name

    def __repr__(self):
        """ Changes Cell object representation to string format to make debugging easier"""
        return self.name

    def add_neighbor(self, neighbor_cell):
        """ Adds neighbor relationship to both connected cells"""
        self.neighbor_list.append(neighbor_cell)
        neighbor_cell.neighbor_list.append(self)

    def get_neighbors(self):
        """ Gets list of neighbor cells to current cell"""
        return self.neighbor_list

    def get_cell_state(self):
        """ Returns state of cell, where 0 = empty, 1 = belong to player 1 and 2 = belong to player 2"""
        return self.cell_state

    def set_cell_state(self, new_state):
        """ Change state of cell, where 0 is empty, 1 belong to player 1 and 2 belong to player 2"""
        if new_state == 0 or new_state == 1 or new_state == 2:
            self.cell_state = new_state
        else:
            raise Exception("Invalid state value. Cell state can only be set to 0, 1 or 2")

    def get_location(self):
        """ Returns location of cell as (row, col)"""
        return self.location


