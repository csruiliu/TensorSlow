from operations import *

_gradient_registry = {}


class RegisterGradient:
    def __init__(self, op_type):
        self._op_type = eval(op_type)

    def __call__(self, f):
        _gradient_registry[self._op_type] = f
        return f


@RegisterGradient("add")
def _add_gradient(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]


@RegisterGradient("log")
def _log_gradient(op, grad):
    x = op.inputs[0]
    return grad / x


@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)


@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad * B, grad * A]


@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T), A.T.dot(grad)]


@RegisterGradient("negative")
def _negative_gradient(op, grad):
    return -grad


@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    A = op.inputs[0]
    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)


@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
    )) * softmax
