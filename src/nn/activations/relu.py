import numpy as np

from src.nn.activations.activation import Activation


class ReLU(Activation):
    """
    ReLU activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        ReLU (Rectified Linear Unit) activation function behaves as follows:
        For each element in the input x, ReLU replaces it with max(x, 0)
        """

        self.input = input
        self.output = np.maximum(input, 0)

        return self.output

    def backward(self, dloss_dout: np.ndarray) -> np.ndarray:
        """
        The derivative of ReLU is 1 for every x where x > 0 and 0 for x <= 0.

        Thus, using the chain rule, the derivative of the loss with respect to the input of the ReLU layer is:
        dL_dx = dL_dout * dx_dout = dL_dout * (input > 0)
        where dL_dout is the derivative of the loss with respect to the output of the ReLU layer and dx_dout is the derivative of the ReLU function.

        * Remember that in backpropagation, we are given the derivative of the loss with respect to the output of the ReLU layer (dL_dout)
          and we need to propagate the gradient through the ReLU function by multiplying it by the derivative of the ReLU function.
        """

        dx_dout = self.input > 0  # Local gradient

        return dloss_dout * dx_dout
