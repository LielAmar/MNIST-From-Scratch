from abc import ABC, abstractmethod

import numpy as np

from src.nn.layers.layer import Layer


class Optimizer(ABC):

    @abstractmethod
    def initialize(self, layer: Layer) -> None:
        pass

    @abstractmethod
    def update(self, layer: Layer, dweights: np.ndarray, dbias: float = None) -> None:
        pass
