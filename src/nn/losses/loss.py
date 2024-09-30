from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class Loss(ABC):

    def __init__(self, reduction: Literal['sum', 'mean'] = 'mean'):
        self.reduction = reduction

        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        self.y_pred = y_pred
        self.y_true = y_true

        result = self._forward()

        if self.reduction == 'mean':
            return np.mean(result)
        elif self.reduction == 'sum':
            return np.sum(result)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

    @abstractmethod
    def _forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        result = self._backward()

        return result

    @abstractmethod
    def _backward(self) -> np.ndarray:
        pass
