from typing import Sequence, Callable, Optional
import jax.numpy as jnp
from flax import nnx


# Default polynomial coefficients approximating ReLU from Boullé et al. (2020)
DEFAULT_RATIONAL_ALPHA = jnp.array([1.1915, 1.5957, 0.5, 0.0218])
DEFAULT_RATIONAL_BETA = jnp.array([2.383, 0.0, 1.0])


class Rational(nnx.Module):
    """Trainable rational function activation layer P(x) / Q(x) in Flax NNX.

    Approximates arbitrary smooth functions with trainable numerator and denominator
    polynomial coefficients evaluated via Horner's scheme (jnp.polyval).
    Initialized by default with coefficients approximating ReLU(x):
        ref: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
             Rational neural networks, NeurIPS (2020) / arXiv:2004.01902.

    Args:
        p_order: Degree of the numerator polynomial P(x) (default: 3).
        q_order: Degree of the denominator polynomial Q(x) (default: 2).
    """

    def __init__(self, p_order: int = 3, q_order: int = 2):
        if p_order == 3 and q_order == 2:
            alpha_init = DEFAULT_RATIONAL_ALPHA
            beta_init = DEFAULT_RATIONAL_BETA
        else:
            alpha_init = jnp.zeros(p_order + 1).at[-2].set(1.0)
            beta_init = jnp.zeros(q_order + 1).at[-1].set(1.0)

        self.alpha = nnx.Param(alpha_init)
        self.beta = nnx.Param(beta_init)

    @property
    def a(self):
        """Alias for numerator coefficients."""
        return self.alpha

    @property
    def b(self):
        """Alias for denominator coefficients."""
        return self.beta

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.polyval(self.alpha[...], x) / jnp.polyval(self.beta[...], x)



class MLP(nnx.Module):
    """Multi Layer Perceptron in Flax NNX for approximating scaling functions.

    Args:
        features: Sequence of layer dimensions including hidden layers and output dimension.
        din: Input feature dimension (default: 1).
        act: Activation function for hidden layers (default: nnx.sigmoid).
        rngs: PRNG stream manager for weight initialization.
    """

    def __init__(
        self,
        features: Sequence[int],
        din: int = 1,
        act: Callable = nnx.sigmoid,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List()
        in_dim = din
        for f in features[:-1]:
            self.layers.append(nnx.Linear(in_dim, f, rngs=rngs))
            in_dim = f
        self.layers.append(nnx.Linear(in_dim, features[-1], rngs=rngs))
        self.act = act

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class RationalMLP(nnx.Module):
    """Multi Layer Perceptron with Rational activation functions in Flax NNX.

    Args:
        features: Sequence of layer dimensions including hidden layers and output dimension.
        din: Input feature dimension (default: 1).
        p_order: Degree of numerator polynomial for Rational activations.
        q_order: Degree of denominator polynomial for Rational activations.
        rngs: PRNG stream manager for weight initialization.
    """

    def __init__(
        self,
        features: Sequence[int],
        din: int = 1,
        p_order: int = 3,
        q_order: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List()
        self.activations = nnx.List()
        in_dim = din
        for f in features[:-1]:
            self.layers.append(nnx.Linear(in_dim, f, rngs=rngs))
            self.activations.append(Rational(p_order, q_order))
            in_dim = f
        self.layers.append(nnx.Linear(in_dim, features[-1], rngs=rngs))


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer, act in zip(self.layers[:-1], self.activations):
            x = act(layer(x))
        return self.layers[-1](x)


class FSSModel(nnx.Module):
    """Unified Finite-Size Scaling Model in Flax NNX.

    Integrates a neural network scaling function and critical parameters
    (e.g., c1, c2, Tc) into a single trainable NNX module.

    Args:
        scaling_fn: Neural network module approximating the universal scaling function F[X].
        n_critical: Number of critical parameters to optimize (e.g., 2 for [c1, Tc] or 3 for [c1, c2, Tc]).
        bijector: Optional function mapping unconstrained raw parameters to physical bounds.
        init_critical: Optional initial values for raw critical parameters (default: zeros).
    """

    def __init__(
        self,
        scaling_fn: nnx.Module,
        n_critical: int = 2,
        bijector: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        *,
        init_critical: Optional[jnp.ndarray] = None,
    ):
        self.scaling_fn = scaling_fn
        if init_critical is None:
            init_critical = jnp.zeros(n_critical)
        self.fss = nnx.Param(init_critical)
        self.bijector = bijector

    @property
    def critical_params(self) -> jnp.ndarray:
        """Physical critical parameters transformed by bijectors."""
        if self.bijector is not None:
            return self.bijector(self.fss[...])
        return self.fss[...]

    def scaled_input(self, Ls: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
        """Compute the scaled tuning parameter X = (T - Tc) * L^c1."""
        crit = self.critical_params
        if len(crit) == 2:
            c1, Tc = crit[0], crit[1]
        else:
            c1, _, Tc = crit[0], crit[1], crit[2]
        return (Ts - Tc) * (Ls ** c1)

    def scaled_target(self, Ls: jnp.ndarray, As: jnp.ndarray) -> jnp.ndarray:
        """Compute the scaled observable Y = A * L^c2 (c2=0 if 2 parameters)."""
        crit = self.critical_params
        if len(crit) == 2:
            return As
        else:
            c2 = crit[1]
            return As * (Ls ** c2)

    def __call__(self, Ls: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
        """Forward pass predicting the scaled observable: F[(T - Tc) * L^c1]."""
        X = self.scaled_input(Ls, Ts)
        return self.scaling_fn(X)