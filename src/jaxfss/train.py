from typing import Callable, Any, Dict, Tuple, Union, Optional
import jax
import jax.numpy as jnp
from flax import nnx
import optax


def MSELoss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((y_true - y_pred) ** 2)


def NLLLoss(y_true: jnp.ndarray, y_pred: jnp.ndarray, var: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Negative log-likelihood loss for Gaussian errors with variance `var`."""
    v = jnp.maximum(var, eps)
    return 0.5 * jnp.mean(jnp.log(v) + (y_true - y_pred) ** 2 / v)


def fit(
    model: nnx.Module,
    loss_fn: Callable[[nnx.Module], jnp.ndarray],
    optimizer: Union[optax.GradientTransformation, nnx.Optimizer, Dict[str, optax.GradientTransformation]],
    steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a Flax NNX model using JIT-compiled optimization.

    Because NNX models are stateful objects, model parameters are updated in-place.

    Args:
        model: The Flax NNX module to optimize (e.g. FSSModel or MLP).
        loss_fn: Scalar loss function accepting `model` as argument.
        optimizer: Either an Optax GradientTransformation, a dictionary of Optax transformations
            (e.g., {"scaling_fn": opt_mlp, "fss": opt_fss}), or an existing nnx.Optimizer.
        steps: Number of optimization iterations.

    Returns:
        losses: Array of loss values across iterations.
        critical_vals: Array of critical parameter histories across iterations.
    """
    # Configure NNX Optimizer
    if isinstance(optimizer, nnx.Optimizer):
        opt = optimizer
    elif isinstance(optimizer, dict):
        # Multi-optimizer support via optax.multi_transform
        pure_params = nnx.as_pure(nnx.state(model, nnx.Param))

        def make_label(path, _):
            for key in optimizer.keys():
                if any(str(p) == key for p in path):
                    return key
            return list(optimizer.keys())[0]

        labels = jax.tree_util.tree_map_with_path(make_label, pure_params)
        optax_tx = optax.multi_transform(optimizer, labels)
        opt = nnx.Optimizer(model, optax_tx, wrt=nnx.Param)
    else:
        opt = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    # Check if model has critical_params
    has_crit = hasattr(model, "critical_params")
    if has_crit:
        init_crit = jnp.asarray(model.critical_params)
        n_crit = len(init_crit)
    elif hasattr(model, "fss"):
        init_crit = jnp.asarray(model.fss[...])
        n_crit = len(init_crit)
    else:
        n_crit = 0

    @nnx.jit(static_argnums=(1, 3))
    def _run_loop(model, loss_fn, opt, steps: int):
        graphdef, state = nnx.split((model, opt))

        def step_fn(i, val):
            state, losses, critical_vals = val
            m, o = nnx.merge(graphdef, state)
            loss, grads = nnx.value_and_grad(loss_fn)(m)
            o.update(m, grads)
            if n_crit > 0:
                crit = m.critical_params if hasattr(m, "critical_params") else m.fss[...]
                critical_vals = critical_vals.at[i].set(crit)
            new_state = nnx.state((m, o))
            losses = losses.at[i].set(loss)
            return new_state, losses, critical_vals

        losses = jnp.zeros(steps)
        critical_vals = jnp.zeros((steps, n_crit))
        state, losses, critical_vals = jax.lax.fori_loop(0, steps, step_fn, (state, losses, critical_vals))
        nnx.update((model, opt), state)
        return losses, critical_vals

    losses, critical_vals = _run_loop(model, loss_fn, opt, steps)
    return losses, critical_vals