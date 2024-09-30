import numpy as np

from src.nn.layers.layer import Layer
from src.nn.nnelement import NNElement
from src.optimizers.optimizer import Optimizer


class Sequential(NNElement):

    def __init__(self, optimizer: Optimizer, sequence: list[NNElement]):
        self.optimizer = optimizer
        self.sequence = sequence

        for nn_element in self.sequence:
            if isinstance(nn_element, Layer):
                self.optimizer.initialize(nn_element)

    def forward(self, input: np.ndarray) -> np.ndarray:
        for nn in self.sequence:
            input = nn(input)

        return input

    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        for nn_element in reversed(self.sequence):
            if isinstance(nn_element, Layer):
                dloss_dout, dloss_dweights, dloss_dbias = nn_element.backward(dloss_dout)

                # Clip gradients of weights and biases before the Optimizer update
                max_grad_norm = 1.0
                np.clip(dloss_dweights, -max_grad_norm, max_grad_norm, out=dloss_dweights)
                if dloss_dbias is not None:
                    np.clip(dloss_dbias, -max_grad_norm, max_grad_norm, out=dloss_dbias)

                self.optimizer.update(nn_element, dloss_dweights, dloss_dbias)
            else:
                dloss_dout = nn_element.backward(dloss_dout)

        return dloss_dout
