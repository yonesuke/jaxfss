from flax import nnx
import jax.numpy as jnp
import optax
import pytest
from jaxfss import MLP, FSSModel, MSELoss, NLLLoss, fit


def test_losses():
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.1, 1.9, 3.2])
    mse = MSELoss(y_true, y_pred)
    assert mse > 0.0
    assert not jnp.isnan(mse)

    var = jnp.array([0.01, 0.01, 0.01])
    nll = NLLLoss(y_true, y_pred, var)
    assert not jnp.isnan(nll)


def test_fit_single_optimizer():
    mlp = MLP(din=1, features=[8, 1], rngs=nnx.Rngs(0))
    model = FSSModel(scaling_fn=mlp, n_critical=2)

    X = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    Y = X ** 2
    Ls = jnp.ones_like(X)

    def loss_fn(m):
        pred = m(Ls, X)
        return MSELoss(Y, pred)

    optimizer = optax.adam(learning_rate=1e-2)
    steps = 30

    losses, critical_vals = fit(model, loss_fn, optimizer, steps)

    assert len(losses) == steps
    assert critical_vals.shape == (steps, 2)
    assert not jnp.any(jnp.isnan(losses))
    assert losses[-1] < losses[0]


def test_fit_multi_optimizer():
    mlp = MLP(din=1, features=[8, 1], rngs=nnx.Rngs(1))
    model = FSSModel(scaling_fn=mlp, n_critical=2)

    X = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    Y = 2.0 * X
    Ls = jnp.ones_like(X)

    def loss_fn(m):
        pred = m(Ls, X)
        return MSELoss(Y, pred)

    optimizer = {
        "scaling_fn": optax.adam(learning_rate=1e-3),
        "fss": optax.adam(learning_rate=1e-2),
    }
    steps = 30

    losses, critical_vals = fit(model, loss_fn, optimizer, steps)

    assert len(losses) == steps
    assert critical_vals.shape == (steps, 2)
    assert not jnp.any(jnp.isnan(losses))
