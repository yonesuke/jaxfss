from flax import nnx
import jax.numpy as jnp
from jaxfss import MLP, Rational, RationalMLP, FSSModel


def test_mlp_forward():
    model = MLP(din=1, features=[16, 16, 1], rngs=nnx.Rngs(0))
    x = jnp.ones((4, 1))
    out = model(x)

    assert out.shape == (4, 1)
    assert not jnp.any(jnp.isnan(out))


def test_rational_forward():
    r = Rational(p_order=3, q_order=2)
    x = jnp.array([-1.0, 0.0, 1.0])
    out = r(x)

    assert out.shape == (3,)
    assert not jnp.any(jnp.isnan(out))


def test_rational_mlp_forward():
    model = RationalMLP(din=1, features=[16, 16, 1], rngs=nnx.Rngs(42))
    x = jnp.ones((4, 1))
    out = model(x)

    assert out.shape == (4, 1)
    assert not jnp.any(jnp.isnan(out))


def test_fss_model_forward():
    mlp = MLP(din=1, features=[8, 1], rngs=nnx.Rngs(0))
    model = FSSModel(scaling_fn=mlp, n_critical=2)

    Ls = jnp.array([16.0, 32.0, 64.0]).reshape(-1, 1)
    Ts = jnp.array([0.4, 0.44, 0.48]).reshape(-1, 1)

    out = model(Ls, Ts)
    assert out.shape == (3, 1)
    assert not jnp.any(jnp.isnan(out))
    assert len(model.critical_params) == 2
