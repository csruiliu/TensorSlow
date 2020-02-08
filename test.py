from graph import Graph, Variable, Placeholder
from operations import *
from session import Session

Graph().as_default()

A = Variable([[1,0],[0,-1]])
b = Variable([1,1])

x = Placeholder()
y = matmul(A, x)
z = add(y, b)

session = Session()
output = session.run(z, {x: [1, 2]})
print(output)