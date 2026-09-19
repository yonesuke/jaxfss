# Usage

Here, we highlight the main building blocks to perform finite-size scaling analysis with `jaxfss` using **Flax NNX**.

---

## 1. Scaling Function (`nnx.Module`)

In `jaxfss`, the unknown universal scaling function $F[X]$ is approximated by an object-oriented neural network module built on [Flax NNX](https://flax.readthedocs.io/).

### Standard MLP
A multilayer perceptron with customizable activations (default: `nnx.sigmoid`):

```python
from flax import nnx
import jaxfss

mlp = jaxfss.MLP(din=1, features=[20, 20, 1], rngs=nnx.Rngs(0))
y = mlp(x)  # Direct object-oriented forward call
```

### RationalMLP
A network with trainable rational function activations ($P(x)/Q(x)$) from [arXiv:2004.01902](https://arxiv.org/abs/2004.01902):

```python
from flax import nnx
import jaxfss

mlp = jaxfss.RationalMLP(din=1, features=[20, 20, 1], rngs=nnx.Rngs(0))
y = mlp(x)
```

`RationalMLP` provides smooth physical function approximation without saturation or vanishing gradient problems.

---

## 2. Unified FSS Model (`FSSModel`)

`FSSModel` encapsulates the neural scaling function, critical exponents ($c_1, c_2$), critical point ($T_{\mathrm{c}}$), and parameter bijectors into a single stateful NNX module.

```python
import jax.numpy as jnp
from flax import nnx
from softclip import SoftClip
import jaxfss

# Define parameter bounds using bijectors
bij_c1 = SoftClip(low=0.0)
bij_c2 = SoftClip(low=0.0)
bij_Tc = SoftClip(low=-1.0, high=1.0)

def bijector(raw_params):
    p1, p2, pc = raw_params
    return jnp.array([
        bij_c1.forward(p1),
        bij_c2.forward(p2),
        bij_Tc.forward(pc),
    ])

scaling_fn = jaxfss.RationalMLP(din=1, features=[20, 20, 1], rngs=nnx.Rngs(0))
model = jaxfss.FSSModel(scaling_fn=scaling_fn, n_critical=3, bijector=bijector)

# Forward pass automatically computes scaled input: F[(T - Tc) * L^c1]
pred = model(Ls, Ts)

# Target scaling: Y = A * L^c2
scaled_Y = model.scaled_target(Ls, As)

# Access physical critical parameters
c1, c2, Tc = model.critical_params
```

---

## 3. Data Handler (`CriticalData`)

Normalizing physical input data before feeding them into neural networks is crucial for numerical stability and convergence.
`CriticalData` automatically applies affine scaling:

```python
import jaxfss

Ls = ...      # System size array
Ts = ...      # Temperature / tuning parameter array
As = ...      # Observable array
As_err = ...  # Error bar array

dataset = jaxfss.CriticalData(Ls, Ts, As, As_err)
train_data = dataset.train_data
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

## 4. Loss Functions

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

## 5. Optimization (`fit`)

`jaxfss.fit` executes a JIT-compiled optimization loop using Flax NNX and Optax.
Because NNX models are stateful, model parameters are updated in-place.

```python
import optax
import jaxfss

# Define loss function on the model
def loss_fn(m: jaxfss.FSSModel):
    y_pred = m(Ls, Ts)
    y_true = m.scaled_target(Ls, As)
    return jaxfss.MSELoss(y_true, y_pred)

# Multi-optimizer: separate learning rates for NN weights and critical exponents
optimizer = {
    "scaling_fn": optax.adam(learning_rate=1e-3),
    "fss": optax.adam(learning_rate=1e-2),
}

steps = 10000
losses, critical_vals = jaxfss.fit(model, loss_fn, optimizer, steps)

# Model parameters are updated in-place:
c1, c2, scaled_Tc = model.critical_params
physical_Tc = dataset.bij_temperature.inverse(scaled_Tc)
```
