from graph import Operation
from graph import Variable
from queue import Queue
from gradients import _gradient_registry


class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)

                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        node.value -= learning_rate * grad

        return MinimizationOperation()

def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0

            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                lossgrad_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrad_wrt_consumer_inputs

                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)

                    lossgrad_wrt_node = lossgrad_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    grad_table[node] += lossgrad_wrt_node

        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table