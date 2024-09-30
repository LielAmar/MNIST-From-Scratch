import numpy as np

from src.nn.activations.activation import Activation


class Sigmoid(Activation):
    """
    Sigmoid activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function behaves as follows:
        For each element in the input x, Sigmoid replaces it with 1 / (1 + exp(-x))
        """

        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        The derivative of Sigmoid is (sigmoid(x) * (1 - sigmoid(x)))

        Thus, using the chain rule, the derivative of the loss with respect to the input of the Sigmoid layer is:
        dL_dx = dL_dout * dx_dout = dL_dout * (sigmoid(x) * (1 - sigmoid(x)))
        where dL_dout is the derivative of the loss with respect to the output of the Sigmoid layer and dx_dout is the derivative of the Sigmoid function.

        * Remember that in backpropagation, we are given the derivative of the loss with respect to the output of the Sigmoid layer (dL_dout)
          and we need to propagate the gradient through the Sigmoid function by multiplying it by the derivative of the Sigmoid function.
        """

        dx_dout = self.output * (1 - self.output)  # Local gradient

        return dL_dout * dx_dout
