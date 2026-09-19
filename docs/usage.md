# Usage

Here, we highlight the main building blocks to perform finite-size scaling analysis with `jaxfss`.

---

## 1. Scaling Function

In `jaxfss`, we approximate the unknown universal scaling function $F[X]$ with a neural network.
We provide modules built on [Flax](https://github.com/google/flax):

### Standard MLP
A multilayer perceptron with customizable activations (default: `sigmoid`):

```python
import jax
import jax.numpy as jnp
import jaxfss

mlp = jaxfss.MLP(features=[20, 20, 1])
mlp_params = mlp.init(jax.random.PRNGKey(0), jnp.array([[1]]))
```

### RationalMLP
A network with trainable rational function activations ($P(x)/Q(x)$) from [arXiv:2004.01902](https://arxiv.org/abs/2004.01902):

```python
import jax
import jax.numpy as jnp
import jaxfss

mlp = jaxfss.RationalMLP(features=[20, 20, 1])
mlp_params = mlp.init(jax.random.PRNGKey(0), jnp.array([[1]]))
```

`RationalMLP` is especially effective for smooth physical functions without vanishing gradient or sharp cusp artifacts.

---

## 2. Data Handler (`CriticalData`)

Normalizing physical input data before feeding them into neural networks is crucial for numerical stability and convergence.
`CriticalData` automatically applies affine scaling:

```python
import jaxfss

Ls = ...      # System size array
Ts = ...      # Temperature / tuning parameter array
As = ...      # Observable array
As_err = ...  # Error bar array

dataset = jaxfss.CriticalData(Ls, Ts, As, As_err)
train_data = dataset.train_data  # Or dataset.training_data
# Keys: "system_size", "temperature", "observable", "observable_var"
```

You can also load directly from a four-column text file ($L$, $T$, $A$, $A_{\mathrm{err}}$):

```python
dataset = jaxfss.CriticalData.from_file(fname="filename.txt")
train_data = dataset.train_data
```

The bijector used for temperature normalization is available as `dataset.bij_temperature` to invert normalized predictions back to physical units:

```python
physical_Tc = dataset.bij_temperature.inverse(scaled_Tc)
```

---

## 3. Loss Functions

`jaxfss` provides loss functions matching common statistical criteria:

### Mean Squared Error (`MSELoss`)
For datasets with uniform or negligible error bars:

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^N (Y_i - \hat{Y}_i)^2
$$

```python
loss = jaxfss.MSELoss(y_true, y_pred)
```

### Negative Log-Likelihood (`NLLLoss`)
For Monte Carlo data with heteroscedastic error bars / variances:

$$
\mathcal{L}_{\mathrm{NLL}} = \frac{1}{2N} \sum_{i=1}^N \left[ \frac{(Y_i - \hat{Y}_i)^2}{\sigma_i^2} + \log(2\pi \sigma_i^2) \right]
$$

```python
loss = jaxfss.NLLLoss(y_true, y_pred, var)
```

---

## 4. Optimization (`fit`)

The `fit` function executes a JIT-compiled optimization loop using [Optax](https://github.com/deepmind/optax).

### Multi-Optimizer (Recommended)
Because neural network weights and physical critical exponents have different scales and sensitivities, using separate learning rates is recommended:

```python
import optax
import jaxfss

init_params = {
    "mlp": mlp_params,
    "fss": jnp.zeros(2)  # Critical parameters
}

# Separate optimizers for MLP and FSS parameters
optimizer = {
    "mlp": optax.adam(learning_rate=1e-3),
    "fss": optax.adam(learning_rate=1e-2),
}

steps = 10000
params, losses, critical_vals = jaxfss.fit(loss_fn, optimizer, init_params, steps)
```

### Single Optimizer
You can also pass a single Optax optimizer:

```python
optimizer = optax.adam(learning_rate=1e-3)
params, losses, critical_vals = jaxfss.fit(loss_fn, optimizer, init_params, steps)
```

`critical_vals` records the history of `params["fss"]` across all optimization steps.
