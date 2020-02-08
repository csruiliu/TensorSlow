from tensorslow.graph import Operation
from tensorslow.graph import Graph
from tensorslow.graph import Variable
from tensorslow.graph import placeholder
from tensorslow.operations import *

Graph().as_default()

A = Variable([[1,0],[0,-1]])
b = Variable([1,1])

x = placeholder()
y = matmul(A, x)
z = add(y, b)

print(z)