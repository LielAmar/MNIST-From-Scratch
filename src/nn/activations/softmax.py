import numpy as np

from src.nn.activations.activation import Activation


class Softmax(Activation):
    """
    Softmax activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Softmax activation function behaves as follows:

        For each element in the input x, Softmax replaces it with exp(x) / sum(exp(x))
        """

        self.input = input

        # Shifting the input by a constant to avoid numerical instability. Works due to Softmax's invariance.
        shifted_input = input - np.max(input, axis=1, keepdims=True)

        exp_values = np.exp(shifted_input)

        exp_sums = np.sum(exp_values, axis=1, keepdims=True)

        self.output = exp_values / exp_sums
        return self.output

    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        """
        Since the Softmax function is used alongside Cross-Entropy loss, and Cross-Entropy loss's derivative
        is already with respect to the logits (input of the Softmax layer), we can directly use the derivative
        of the loss with respect to the input of the Softmax layer.

        Thus, the derivative of the loss with respect to the input of the Softmax layer is:
        dL_dx = dL_dout
        """

        return dloss_dout
