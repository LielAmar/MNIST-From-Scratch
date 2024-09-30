from typing import Literal

import numpy as np

from src.losses.loss import Loss


class MeanSquaredError(Loss):
    """
    Mean Squared Error losses function
    """

    def __init__(self, reduction: Literal['sum', 'mean'] = 'mean'):
        super().__init__(reduction)

    def _forward(self) -> np.ndarray:
        """
        The Mean Squared Error is computed as follows:

        MeanSquaredError = \Sum_{i=1}^{n} (y_pred_i - y_true_i) ^ 2
        Or, in vector notations: CrossEntropyLoss = (y_pred - y_true) ^ 2
        """

        return (self.y_pred - self.y_true) ** 2

    def _backward(self) -> np.ndarray:
        """
        The derivative of the MeanSquaredError with respect to the predictions is as follows:

        (\grad MeanSquaredError) / (\grad y_pred_i) = (\grad (\Sum_{i=1}^{n} (y_pred_i - y_true_i) ^ 2)) / (\grad y_pred_i).
        We know that for each i, the derivative of all the items in the sum with respect to y_pred_i are going to be 0, expect for the i'th item.

        Thus, (\grad MeanSquaredError) / (\grad y_pred_i) = (\grad (y_pred_i - y_true_i) ^ 2) / (\grad y_pred_i) =
            = 2 * (y_pred_i - y_true_i)
        """

        return 2 * (self.y_pred - self.y_true)
