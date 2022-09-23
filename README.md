# jaxfss
JAX/Flax implementation of finite-size scaling

## Installation
`jaxfss` can be installed with pip with the following command:
```
pip install jaxfss
```

## Quickstart
`jaxfss` is a tool for finite-size scaling.
See finite-size scaling on the Binder ratio of the 2d Ising model in [examples](examples).

## Components
Here, we highlight the main blocks to do the finite-size scaling.
### `MLP` and `RationalMLP`
In jaxfss, we approximate the scaling function to a neural netowrk.
We provide a simple multi layer perceptron (MLP) module to construct neural networks, which is build upon [Flax](https://github.com/google/flax).
```python
import jaxfss
import jax
import jax.numpy as jnp

mlp = jaxfss.MLP(features=[20,20,1])
mlp_params = mlp.init(jax.random.PRNGKey(0), jnp.array([[1]]))
```
Default activation function of `MLP` is a sigmoid function. You can change this via `act` argument.

We also provide the MLP with rational activation function.
```python
import jaxfss
import jax.numpy as jnp
from jax import random

mlp = jaxfss.RationalMLP(features=[20,20,1])
mlp_params = mlp.init(jax.random.PRNGKey(0), jnp.array([[1]]))
```
In `RationalMLP`, the activation function is approximated by a rational function,
and the parameters of the activation function are also optimized during the learning process.

See [arXiv:2004.01902](https://arxiv.org/abs/2004.01902) for the original paper.

### `CriticalData`
In neural networks, it is very important to pre-normalize the training data.
`CriticalData` is a module that does the pre-processing for you.

```python
import jaxfss

Ls = ... # systemsize arrays
Ts = ... # temperature arrays
As = ... # observable arrays
As_err = ... # observable error arrays
dataset = jaxfss.CriticalData(Ls, Ts, As, As_err)
train_data = dataset.training_data # dict with "system_size", "temperature", "observable", "observable_var"
```
You can also initialize the module from file:
```python
import jaxfss

dataset = jaxfss.CriticalData.from_file(fname="filename.txt")
train_data = dataset.training_data
```

### `MSELoss` and `NLLLoss`
We provide a helper function for constructing loss function.
`MSELoss` is the mean squared error, and `NLLLoss` is the negative log likelihood loss.
The function name is derived from pytorch.
```python
import jaxfss

def loss_fn(params):
    ...
    y_true = ...
    y_pred = ...
    return jaxfss.MSELoss(y_true, y_pred)
```

### `fit`
Once everything is ready, all you need to do is learn!
`fit` function is a wrapper that learns using [optax](https://github.com/deepmind/optax).
We note that `init_params` should be a dict with keys `mlp` and `fss`.

```python
import jaxfss
import optax

init_params = {
    "mlp": mlp_params,
    "fss": ...
}
def loss_fn(params):
    ...
    return ...
optimizer = optax.adam(learning_rate=10**-3)
steps = 10**4
params, losses, fsses = jaxfss.fit(loss_fn, optimizer, init_params, steps)
```
`fit` function returns the final `params` together with values of loss and fss in the learning process.
## Further reading
- [FSS method by GP with C++](https://kenjiharada.github.io/BSA/)
- [FSS method by NN and GP with Python (Pytorch)](https://github.com/KenjiHarada/FSS-tools)
## Citation
We are currently preparing for the paper!!
