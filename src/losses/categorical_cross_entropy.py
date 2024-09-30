from typing import Literal

import numpy as np

from src.losses.loss import Loss


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy losses function
    """

    def __init__(self, reduction: Literal['sum', 'mean'] = 'sum', from_logits: bool = True):
        super().__init__(reduction)

        self.from_logits = from_logits

    def _forward(self) -> np.ndarray:
        """
        The Cross Entropy Loss is computed as follows:

        CrossEntropyLoss = \Sum_{j=1}^{c} \Sum_{i=1}^n (-y_true_i * log(y_pred_i))
        Or, in vector notations: CrossEntropyLoss = \Sum_{j=1}^{c} (-y_true * log(y_pred))
        """

        if self.from_logits:
            shifted_logits = self.y_pred - np.max(self.y_pred, axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            self.y_pred = softmax_probs

        epsilon = 1e-12  # Clipping to avoid log(0)
        clipped_y_pred = np.clip(self.y_pred, epsilon, 1. - epsilon)

        return -np.sum(self.y_true * np.log(clipped_y_pred), axis=-1)

    def _backward(self) -> np.ndarray:
        """
        The derivative of the CrossEntropyLoss with respect to the predictions is as follows (assuming the correct class is i):

        # (a)
        (\grad CrossEntropyLoss) / (\grad y_pred_i) = \Sum_{j!=i} (0) + -y_true_i/y_pred_i
        Essentially, since we have one-hot representation, we know that y_true_j is 0 for every j != i.

        Thus, the gradient is -y_true_i / y_pred_i.
        We can further simplify it using the chain rule.

        Since we use Softmax, we know that y_pred_i = e^{z_i} / Sum_{j=1}^n (e^{z_j}).
        where z_j is the output of the last linear layer (before softmax) for every j. Also known as logits.

        We can use this to compute (\grad CrossEntropyLoss) / (\grad z_i) (the derivative of the loss with respect to the logits).

        (\grad CrossEntropyLoss) / (\grad z_i) = \Sum_k ((\grad CrossEntropyLoss) / (\grad y_pred_k)) * ((\grad pred_y_k) / (\grad z_i))
                = \Sum_k (-y_true_k/y_pred_k) * ((\grad pred_y_k) / (\grad z_i))  # Using (a)

        We also know that (\grad pred_y_k) / (\grad z_i) = y_pred_k * (1 - y_pred_k) if i = k, and -y_pred_k * y_pred_i otherwise.

        Finally, ignoring cases where i != k, we get that:
        (\grad CrossEntropyLoss) / (\grad z_i) = (-y_true_i / y_pred_i) * (y_pred_i * (1 - y_pred_i)) = (-y_true_i) * (1 - y_pred_i).
        and since y_true_i = 1, we get that:

        (\grad CrossEntropyLoss) / (\grad z_i) = y_pred_i - y_true_i = y_pred_i - 1
        """

        return self.y_pred - self.y_true
