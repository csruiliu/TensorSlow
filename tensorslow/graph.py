class Graph:
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self


class Operation:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(self)
        
        _default_graph.operations.append(self)

    def compute(self):
        pass

class placeholder:
    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)


class Variable:    
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []
        _default_graph.variables.append(self)