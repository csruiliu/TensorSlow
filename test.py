from graph import Graph, Variable, Placeholder
from operations import *
from session import Session
from train import GradientDescentOptimizer

def test_compute_graph():
    Graph().as_default()

    A = Variable([[1, 0], [0, -1]])
    b = Variable([1, 1])

    x = Placeholder()
    y = matmul(A, x)
    z = add(y, b)

    session = Session()
    output = session.run(z, {x: [1, 2]})
    print(output)

def test_perceptron():
    Graph().as_default()
    x = Placeholder()
    w = Variable([1, 1])
    b = Variable(0)
    p = sigmoid(add(matmul(w, x), b))

    session = Session()
    output = session.run(p, {x: [3, 2]})
    print(output)

def test_perceptron_loss():
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    Graph().as_default()
    X = Placeholder()
    c = Placeholder()

    W = Variable([[1, -1],[1, -1]])

    b = Variable([0, 0])
    p = softmax(add(matmul(X, W), b))
    J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

    session = Session()
    print(session.run(J, {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }))

def test_train():
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))
    Graph().as_default()

    X = Placeholder()
    c = Placeholder()

    # Initialize weights randomly
    W = Variable(np.random.randn(2, 2))
    b = Variable(np.random.randn(2))

    # Build perceptron
    p = softmax(add(matmul(X, W), b))

    # Build cross-entropy loss
    J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

    # Build minimization op
    minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)

    # Build placeholder inputs
    feed_dict = {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }

    # Create session
    session = Session()

    # Perform 100 gradient descent steps
    for step in range(100):
        J_value = session.run(J, feed_dict)
        if step % 10 == 0:
            print("Step:", step, " Loss:", J_value)
        session.run(minimization_op, feed_dict)

    # Print final result
    W_value = session.run(W)
    print("Weight matrix:\n", W_value)
    b_value = session.run(b)
    print("Bias:\n", b_value)

if __name__ == "__main__":
    #test_compute_graph()
    #test_perceptron()
    #test_perceptron_loss()
    test_train()