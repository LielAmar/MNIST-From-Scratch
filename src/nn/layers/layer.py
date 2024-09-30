from abc import abstractmethod

import numpy as np

from src.nn.nnelement import NNElement


class Layer(NNElement):

    def __init__(self, include_bias: bool = True):
        self.weights = None
        self.bias = None

        self.input = None
        self.output = None

        self.include_bias = include_bias

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        pass
