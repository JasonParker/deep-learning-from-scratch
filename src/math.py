import numpy as numpy


def derivative(func: Callable[[ndarray], ndarray],
               input_: ndarray,
               delta: float = 0.001) -> ndarray:
    '''
    Evaluatves the derivative of a function "func" at every element in 
    the "input_" array.
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray.

    For additional context on Leaky ReLU:
    https://paperswithcode.com/method/leaky-relu
    '''
    return np.maximum(0.2 * x, x)


def relu(x: ndarray) -> ndarray:
    '''
    Apply "ReLU" function to each element in ndarray.

    For additional context on ReLU:
    https://paperswithcode.com/method/relu
    '''
    return np.maximum(x, 0)


def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
    return 1 / (1 + np.exp(-x))


def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''
    return np.power(x, 2)



