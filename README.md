# jaxfss

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxfss)
![PyPI](https://img.shields.io/pypi/v/jaxfss)
[![Test](https://github.com/yonesuke/jaxfss/actions/workflows/test.yml/badge.svg)](https://github.com/yonesuke/jaxfss/actions/workflows/test.yml)
[![Documentation](https://github.com/yonesuke/jaxfss/actions/workflows/book.yml/badge.svg)](https://yonesuke.github.io/jaxfss/)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://yonesuke.github.io/jaxfss/)
[![Downloads](https://static.pepy.tech/badge/jaxfss)](https://pepy.tech/project/jaxfss)

**Flax NNX implementation of Neural Scaling Analysis (NSA) for Finite-Size Scaling**

![ising_binder](https://user-images.githubusercontent.com/12659790/191948671-dc28959d-0e24-4197-baca-0c2ef0aad311.png)

Near a continuous phase transition, physical quantities in a finite-size system obey the finite-size scaling law:

$$
A(T, L) = L^{-c_2} F\left[ (T - T_{\mathrm{c}}) L^{c_1} \right]
$$

where $A(T, L)$ is an observable at temperature $T$ in a finite-size system of linear dimension $L$, $T_{\mathrm{c}}$ is the critical point, $c_1$ and $c_2$ are critical exponents, and $F[\cdot]$ is the unknown universal scaling function.

`jaxfss` estimates critical parameters $(T_{\mathrm{c}}, c_1, c_2)$ and the scaling function $F$ simultaneously using **Flax NNX** (the next-generation object-oriented API for Flax) and JIT-compiled optimization with JAX and Optax.

## Key Features

- **Flax NNX native**: Modern object-oriented architecture (`model = FSSModel(...)`).
- **Linear computational complexity**: $\mathcal{O}(N)$ scaling with respect to data points, scaling smoothly to large-scale Monte Carlo datasets.
- **Built-in Rational activations**: Trainable `RationalMLP` for smooth, high-fidelity physical function representation without external dependencies.
- **Unified FSS model**: `FSSModel` encapsulates neural scaling functions, critical exponents, and parameter bijectors.
- **Multi-optimizer training**: Separate learning rates for neural network weights and physical critical exponents.
- **Hardware accelerated**: Fast training via JAX JIT compilation on CPU/GPU/TPU.

## Installation

```bash
pip install jaxfss
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add jaxfss
```

## Quickstart

```python
from flax import nnx
import optax
from softclip import SoftClip
import jaxfss

# 1. Load data
dataset = jaxfss.CriticalData.from_file("ising.txt")
Ls, Ts, As = dataset.train_data["system_size"], dataset.train_data["temperature"], dataset.train_data["observable"]

# 2. Build NNX scaling model
bij_c1 = SoftClip(low=0.0)
bij_Tc = SoftClip(low=-1.0, high=1.0)
bijector = lambda p: [bij_c1.forward(p[0]), bij_Tc.forward(p[1])]

scaling_fn = jaxfss.RationalMLP(din=1, features=[20, 20, 1], rngs=nnx.Rngs(0))
model = jaxfss.FSSModel(scaling_fn=scaling_fn, n_critical=2, bijector=bijector)

# 3. Optimize
loss_fn = lambda m: jaxfss.MSELoss(As, m(Ls, Ts))
optimizer = {
    "scaling_fn": optax.adam(1e-3),
    "fss": optax.adam(1e-2),
}

losses, critical_vals = jaxfss.fit(model, loss_fn, optimizer, steps=8000)

c1, scaled_Tc = model.critical_params
Tc = dataset.bij_temperature.inverse(scaled_Tc)
print(f"c1: {c1:.4f}, Tc: {Tc:.4f}")
```

Check out the full [Documentation and Tutorials](https://yonesuke.github.io/jaxfss/)!

## Related Packages

- [BSA (C++)](https://kenjiharada.github.io/BSA/): Finite-size scaling with Gaussian process regression.
- [FSS-tools (PyTorch)](https://github.com/KenjiHarada/FSS-tools): Finite-size scaling with neural networks and Gaussian processes.

## Citation

Please cite our paper published in *Physical Review E* when you use `jaxfss` in your research:

- Ryosuke Yoneda and Kenji Harada, *Neural network approach to scaling analysis of critical phenomena*, **Phys. Rev. E 107, 044128 (2023)**. [doi:10.1103/PhysRevE.107.044128](https://doi.org/10.1103/PhysRevE.107.044128)

```bibtex
@article{PhysRevE.107.044128,
  title = {Neural network approach to scaling analysis of critical phenomena},
  author = {Yoneda, Ryosuke and Harada, Kenji},
  journal = {Phys. Rev. E},
  volume = {107},
  issue = {4},
  pages = {044128},
  numpages = {10},
  year = {2023},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.107.044128},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.107.044128}
}
```
