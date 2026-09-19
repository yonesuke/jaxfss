from typing import Callable, Any, Dict, Tuple, Union
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.lax import fori_loop
import optax


def MSELoss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((y_true - y_pred) ** 2)


def NLLLoss(y_true: jnp.ndarray, y_pred: jnp.ndarray, var: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Negative log-likelihood loss for Gaussian errors with variance `var`."""
    v = jnp.maximum(var, eps)
    return 0.5 * jnp.mean(jnp.log(v) + (y_true - y_pred) ** 2 / v)


def fit(
    loss_fn: Callable,
    optimizer: Union[optax.GradientTransformation, Dict[str, optax.GradientTransformation]],
    init_params: Any,
    steps: int,
) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """Fit model parameters using Optax optimizer(s).

    Args:
        loss_fn: Loss function that takes params and returns a scalar loss.
        optimizer: Either a single Optax optimizer, or a dict of optimizers
            (e.g., {"mlp": opt_mlp, "fss": opt_fss}) for multi-transform optimization.
        init_params: Initial parameters (dict or tuple/list). If dict, it usually contains
            "mlp" and "fss" keys.
        steps: Number of optimization steps.

    Returns:
        params: Final parameters after optimization.
        losses: Array of loss values at each step.
        critical_vals: Array of FSS parameter values at each step.
    """
    # Configure optimizer
    if isinstance(optimizer, dict):
        if isinstance(init_params, dict):
            param_labels = {k: jax.tree_util.tree_map(lambda _: k, v) for k, v in init_params.items()}
            opt = optax.multi_transform(optimizer, param_labels)
        elif isinstance(init_params, (tuple, list)):
            param_labels = tuple(
                jax.tree_util.tree_map(lambda _, k=k: k, init_params[i])
                for i, k in enumerate(optimizer.keys())
            )
            opt = optax.multi_transform(optimizer, param_labels)
        else:
            raise ValueError("When optimizer is a dict, init_params must be a dict or tuple/list.")
    else:
        opt = optimizer

    opt_state = opt.init(init_params)

    # Determine how to extract FSS parameters
    is_dict = isinstance(init_params, dict) and "fss" in init_params
    if is_dict:
        n_critical = len(init_params["fss"])
        extract_fss = lambda p: p["fss"]
    elif isinstance(init_params, (tuple, list)) and len(init_params) > 1:
        n_critical = len(init_params[1])
        extract_fss = lambda p: p[1]
    else:
        n_critical = 0
        extract_fss = lambda _: jnp.array([])

    @jit
    def update_fn(i, val):
        params, opt_state, losses, critical_vals = val
        loss, grad = value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses = losses.at[i].set(loss)
        if n_critical > 0:
            critical_vals = critical_vals.at[i].set(extract_fss(params))
        return [params, opt_state, losses, critical_vals]

    losses = jnp.zeros(steps)
    critical_vals = jnp.zeros((steps, n_critical))
    init_val = [init_params, opt_state, losses, critical_vals]

    params, opt_state, losses, critical_vals = fori_loop(0, steps, update_fn, init_val)

    return params, losses, critical_vals