import numpy as np

from src.nn.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def initialize(self, layer):
        """
        Initialize the optimizer's state for the given layer.
        """

        self.m[layer] = {'weights': np.zeros_like(layer.weights)}
        self.v[layer] = {'weights': np.zeros_like(layer.weights)}

        if layer.include_bias:
            self.m[layer]['bias'] = np.zeros_like(layer.bias)
            self.v[layer]['bias'] = np.zeros_like(layer.bias)

    def update(self, layer, dW, db=None):
        self.t += 1

        # Update first moment estimate (mean)
        self.m[layer]['weights'] = self.beta_1 * self.m[layer]['weights'] + (1 - self.beta_1) * dW
        if db is not None:
            self.m[layer]['bias'] = self.beta_1 * self.m[layer]['bias'] + (1 - self.beta_1) * db

        # Update second moment estimate (variance)
        self.v[layer]['weights'] = self.beta_2 * self.v[layer]['weights'] + (1 - self.beta_2) * (dW ** 2)
        if db is not None:
            self.v[layer]['bias'] = self.beta_2 * self.v[layer]['bias'] + (1 - self.beta_2) * (db ** 2)

        # Correct bias in first moment
        m_hat_weights = self.m[layer]['weights'] / (1 - self.beta_1 ** self.t)
        if db is not None:
            m_hat_bias = self.m[layer]['bias'] / (1 - self.beta_1 ** self.t)

        # Correct bias in second moment
        v_hat_weights = self.v[layer]['weights'] / (1 - self.beta_2 ** self.t)
        if db is not None:
            v_hat_bias = self.v[layer]['bias'] / (1 - self.beta_2 ** self.t)

        # Update weights and biases using Adam rule
        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        if db is not None:
            layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
