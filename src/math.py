import numpy as np
from numpy import ndarray
from typing import Callable


def binary_step_function(x: ndarray,
                         threshold: float) -> ndarray:
    '''
    Apply binary step function to each element in ndarray using
    float threshold as the basis for activation.
    '''
    def binary_step(item):
        if item < threshold:
            return 0
        else:
            return 1
    return np.array([binary_step(item) for item in x])


def derivative(func: Callable[[ndarray], ndarray],
               input_: ndarray,
               delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in 
    the "input_" array.
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def derivative_chain_rule(chain: List,
                          input_range: ndarray) -> ndarray:
    
    outputs = []
    for i in range(0, len(chain) - 1):
        if i == 0:
            outputs.append(chain[i](input_range))
        else: outputs.append(chain[i](outputs[i - 1]))
        
    derivatives = []
    for i in range(len(chain) - 1, -1 , -1):
        if i == 0:
            derivatives.append(derivative(chain[i], input_range))
        else: derivatives.append(derivative(chain[i], outputs[i - 1]))
        
    for i in range(1, len(chain)):
        if i == 1:
            product = np.multiply(derivatives[i - 1], derivatives[i])
        else:
            product = np.multiply(product, derivatives[i])
    
    return product


def elu_function(x: ndarray, slope: float) -> ndarray:
    '''
    Apply exponential linear unit function to each element in ndarray using
    float slope as the basis for activation.
    '''
    def elu(x, a):
        if x<0:
            return a*(np.exp(x)-1)
        else:
            return x
    return np.array([elu(item, slope) for item in x])


def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray.

    For additional context on Leaky ReLU:
    https://paperswithcode.com/method/leaky-relu
    '''
    return np.maximum(0.2 * x, x)


def linear_activation(x: ndarray, slope: float) -> ndarray:
    '''
    Apply linear function to each element in ndarray using
    float slope as the basis for activation.
    '''
    return x * slope


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


def softmax_function(x: ndarray) -> ndarray:
    '''
    Apply softmax function
    '''
    z = np.exp(x)
    return z/z.sum()


def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''
    return np.power(x, 2)


def swish_function(x: ndarray) -> ndarray:
    '''
    Apply swish function
    '''
    return x/(1-np.exp(-x))


def tanh_function(x: ndarray) -> ndarray:
    '''
    Apply the tanh function to each element in the input ndarray.
    '''
    return (2/(1 + np.exp(-2*x))) -1



