import numpy as np
from graph import Placeholder, Variable, Operation


class Session():

    def run(self, operation, feed_dict={}):
        nodes_postorder = self.traverse_postorder(operation)

        # iterate node in postorder
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output 

    def traverse_postorder(self, operation):
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)

        recurse(operation)

        return nodes_postorder