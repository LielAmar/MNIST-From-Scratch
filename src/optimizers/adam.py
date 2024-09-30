import numpy as np

from src.nn.layers.layer import Layer
from src.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, betas: tuple[float] = (0.9, 0.999), epsilon: float = 1e-08,
                 weight_decay=0):
        self.learning_rate = learning_rate
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def initialize(self, layer: Layer) -> None:
        self.m[layer] = {'weights': np.zeros_like(layer.weights)}
        self.v[layer] = {'weights': np.zeros_like(layer.weights)}

        if layer.include_bias:
            self.m[layer]['bias'] = np.zeros_like(layer.bias)
            self.v[layer]['bias'] = np.zeros_like(layer.bias)

    def update(self, layer: Layer, dweights: np.ndarray, dbias: float = None):
        self.t += 1

        if self.weight_decay != 0:
            dweights += self.weight_decay * layer.weights

        # Weights: Update first moment estimate (mean), second moment estimate (variance), and bias correction both
        self.m[layer]['weights'] = self.betas[0] * self.m[layer]['weights'] + (1 - self.betas[0]) * dweights
        self.v[layer]['weights'] = self.betas[1] * self.v[layer]['weights'] + (1 - self.betas[1]) * (dweights ** 2)

        m_hat_weights = self.m[layer]['weights'] / (1 - self.betas[0] ** self.t)
        v_hat_weights = self.v[layer]['weights'] / (1 - self.betas[1] ** self.t)

        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        # Bias: Update first moment estimate (mean), second moment estimate (variance), and bias correction both
        if dbias is not None:
            self.m[layer]['bias'] = self.betas[0] * self.m[layer]['bias'] + (1 - self.betas[0]) * dbias
            self.v[layer]['bias'] = self.betas[1] * self.v[layer]['bias'] + (1 - self.betas[1]) * (dbias ** 2)

            m_hat_bias = self.m[layer]['bias'] / (1 - self.betas[0] ** self.t)
            v_hat_bias = self.v[layer]['bias'] / (1 - self.betas[1] ** self.t)

            layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
