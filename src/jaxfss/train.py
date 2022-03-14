import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.lax import fori_loop
import optax

def MSELoss(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def NLLLoss(y_true, y_pred, var, eps=10**-6):
    v = jnp.maximum(var, eps)
    return 0.5 * jnp.mean(jnp.log(v) + (y_true - y_pred) ** 2 / v)

def fit(loss_fn, optimizer, init_params, steps):
    opt_state = optimizer.init(init_params)

    @jit
    def updata_fn(i, val):
        params, opt_state, losses, critical_vals = val
        loss, grad = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses = losses.at[i].set(loss)
        critical_vals = critical_vals.at[i].set(params["fss"])
        return [params, opt_state, losses, critical_vals]

    losses = jnp.zeros(steps)
    n_critical = len(init_params["fss"])
    critical_vals = jnp.zeros((steps, n_critical))
    init_val = [init_params, opt_state, losses, critical_vals]

    params, opt_state, losses, critical_vals = fori_loop(0, steps, updata_fn, init_val)

    return params, losses, critical_vals