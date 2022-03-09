from flax import linen as nn
from typing import Sequence, Callable

from rationalnets import RationalMLP

class MLP(nn.Module):
    """
    Multi layer perceptron
    """
    features: Sequence[int]
    act: Callable = nn.sigmoid
    
    @nn.compact
    def __call__(self, x):
        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = self.act(x)
        x = nn.Dense(self.features[-1])(x)
        return x