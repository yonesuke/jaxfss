from typing import Sequence, Callable
import jax.numpy as jnp
from flax import linen as nn

from rationalnets import RationalMLP


class MLP(nn.Module):
    """Multi Layer Perceptron for approximating scaling functions.

    Attributes:
        features: Sequence of layer dimensions including intermediate and output layers.
        act: Activation function for hidden layers (defaults to nn.sigmoid).
    """

    features: Sequence[int]
    act: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = self.act(x)
        x = nn.Dense(self.features[-1])(x)
        return x