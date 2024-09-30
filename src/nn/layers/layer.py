from abc import abstractmethod

import numpy as np

from src.nn.nnelement import NNElement


class Layer(NNElement):

    def __init__(self):
        self.weights = None
        self.bias = None

        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        pass
