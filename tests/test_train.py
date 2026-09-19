import jax
import jax.numpy as jnp
import optax
import pytest
from jaxfss import MLP, MSELoss, NLLLoss, fit


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
    key = jax.random.PRNGKey(0)
    model = MLP(features=[8, 1])
    mlp_params = model.init(key, jnp.ones((1, 1)))

    init_params = {
        "mlp": mlp_params,
        "fss": jnp.array([0.5, 0.5]),
    }

    # Dummy target
    X = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    Y = X ** 2

    def loss_fn(params):
        c = params["fss"][0]
        pred = model.apply(params["mlp"], X) * c
        return MSELoss(Y, pred)

    optimizer = optax.adam(learning_rate=1e-2)
    steps = 20

    params, losses, critical_vals = fit(loss_fn, optimizer, init_params, steps)

    assert len(losses) == steps
    assert critical_vals.shape == (steps, 2)
    assert not jnp.any(jnp.isnan(losses))
    assert losses[-1] < losses[0]


def test_fit_multi_optimizer():
    key = jax.random.PRNGKey(1)
    model = MLP(features=[8, 1])
    mlp_params = model.init(key, jnp.ones((1, 1)))

    init_params = {
        "mlp": mlp_params,
        "fss": jnp.array([0.5, 0.5]),
    }

    X = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    Y = 2.0 * X

    def loss_fn(params):
        c = params["fss"][0]
        pred = model.apply(params["mlp"], X) + c
        return MSELoss(Y, pred)

    optimizer = {
        "mlp": optax.adam(learning_rate=1e-3),
        "fss": optax.adam(learning_rate=1e-2),
    }
    steps = 20

    params, losses, critical_vals = fit(loss_fn, optimizer, init_params, steps)

    assert len(losses) == steps
    assert critical_vals.shape == (steps, 2)
    assert not jnp.any(jnp.isnan(losses))
