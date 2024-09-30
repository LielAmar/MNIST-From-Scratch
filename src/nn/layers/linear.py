import numpy as np

from src.nn.layers.layer import Layer


class Linear(Layer):
    def __init__(self, input_dimension: int, output_dimension: int, include_bias: bool = True):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.include_bias = include_bias

        # Starting off with random weights and random bias
        # Reference: https://medium.com/@Coursesteach/deep-learning-part-27-random-initialization-b25ef8df8334
        # "Section 2 - Random Initialization"
        self.weights = np.random.rand(self.output_dimension, self.input_dimension) * 0.01

        if include_bias:
            self.bias = np.zeros(shape=(self.output_dimension,))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input

        self.output = input @ self.weights.T

        if self.include_bias:
            self.output = self.output + self.bias

        return self.output

    def backward(self, dL_dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for the Linear layer.
        It needs to compute 3 things:
        1. The gradient of the loss with respect to the weights of the Linear layer. This is needed to update the weights of the Linear layer.
        2. The gradient of the loss with respect to the bias of the Linear layer. This is needed to update the bias of the Linear layer.
        3. The gradient of the loss with respect to the input of the Linear layer. This is needed to propagate the gradient through the Linear layer.

        Once again, we use the chain rule to compute these gradients.
        Let's mark the loss as L, the output of the Linear layer as Y, the weights of the Linear layer as W,
        the biases of the Linear layer as b and the input of the Linear layer as X.

        1. (\partial L / \partial W) = \partial L / \partial Y * \partial Y / \partial W = dL_out * (\partial Y / \partial W)
            = dL_out * (\partial (X @ W.T) / \partial W) = dL_out * X
        2. (\partial L / \partial b) = \partial L / \partial Y * \partial Y / \partial b = dL_out * (\partial Y / \partial b)
            = dL_out * (\partial (X @ W.T + b) / \partial b) = dL_out * 1
        3. (\partial L / \partial X) = \partial L / \partial Y * \partial Y / \partial X = dL_out * (\partial Y / \partial X)
            = dL_out * (\partial (X @ W.T) / \partial X) = dL_out * W
        """

        # Gradient of the loss with respect to the weights
        dL_dW = dL_dout.T @ self.input

        # Gradient of the loss with respect to the bias (if bias is included)
        # We sum the gradients of all the samples in the batch because we added the bias to every sample, and they all contributed to the loss.
        dL_db = np.sum(dL_dout, axis=0) if self.include_bias else None

        # Gradient of the loss with respect to the input
        dL_dinput = dL_dout @ self.weights

        return dL_dinput, dL_dW, dL_db
