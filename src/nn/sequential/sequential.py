import numpy as np

from src.nn.nnelement import NNElement


class Sequential:
    def __init__(self, sequence: list[NNElement]):
        self.sequence = sequence

    def forward(self, input: np.ndarray) -> np.ndarray:
        for nn in self.sequence:
            input = nn(input)

        return input
