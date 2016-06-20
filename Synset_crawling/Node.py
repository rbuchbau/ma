class Node:
    name = ''
    id = -1
    children = None

    def __init__(self):
        children = []

    def toString(self):
        return self.name + " " + str(self.id) + " " + str(self.children)
