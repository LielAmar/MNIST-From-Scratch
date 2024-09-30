from abc import abstractmethod

import numpy as np

from src.nn.nnelement import NNElement


class Activation(NNElement):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        pass
