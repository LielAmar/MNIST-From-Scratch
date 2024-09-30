from abc import ABC, abstractmethod

import numpy as np


class NNElement(ABC):
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)
