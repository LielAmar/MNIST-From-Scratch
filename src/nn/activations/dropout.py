import numpy as np

from src.nn.activations.activation import Activation


class Dropout(Activation):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.mask = (np.random.rand(*input.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
        return input * self.mask

    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        return dloss_dout * self.mask
