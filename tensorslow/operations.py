from tensorslow.graph import Operation

# add class inhert Operation
class add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value + y_value

class matmul(Operation):
    
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)