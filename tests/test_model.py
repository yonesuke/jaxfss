import jax
import jax.numpy as jnp
from jaxfss import MLP, RationalMLP


def test_mlp_forward():
    key = jax.random.PRNGKey(0)
    model = MLP(features=[16, 16, 1])
    x = jnp.ones((4, 1))
    params = model.init(key, x)
    out = model.apply(params, x)

    assert out.shape == (4, 1)
    assert not jnp.any(jnp.isnan(out))


def test_rational_mlp_forward():
    key = jax.random.PRNGKey(42)
    model = RationalMLP(features=[16, 16, 1])
    x = jnp.ones((4, 1))
    params = model.init(key, x)
    out = model.apply(params, x)

    assert out.shape == (4, 1)
    assert not jnp.any(jnp.isnan(out))
