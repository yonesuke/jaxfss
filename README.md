# jaxfss

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxfss)
![PyPI](https://img.shields.io/pypi/v/jaxfss)
[![Test](https://github.com/yonesuke/jaxfss/actions/workflows/test.yml/badge.svg)](https://github.com/yonesuke/jaxfss/actions/workflows/test.yml)
[![Documentation](https://github.com/yonesuke/jaxfss/actions/workflows/book.yml/badge.svg)](https://yonesuke.github.io/jaxfss/)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://yonesuke.github.io/jaxfss/)
[![Downloads](https://static.pepy.tech/badge/jaxfss)](https://pepy.tech/project/jaxfss)

**JAX/Flax implementation of Neural Scaling Analysis (NSA) for Finite-Size Scaling**

![ising_binder](https://user-images.githubusercontent.com/12659790/191948671-dc28959d-0e24-4197-baca-0c2ef0aad311.png)

Near a continuous phase transition, physical quantities in a finite-size system obey the finite-size scaling law:

$$
A(T, L) = L^{-c_2} F\left[ (T - T_{\mathrm{c}}) L^{c_1} \right]
$$

where $A(T, L)$ is an observable at temperature $T$ in a finite-size system of linear dimension $L$, $T_{\mathrm{c}}$ is the critical point, $c_1$ and $c_2$ are critical exponents, and $F[\cdot]$ is the unknown universal scaling function.

`jaxfss` estimates critical parameters $(T_{\mathrm{c}}, c_1, c_2)$ and the scaling function $F$ simultaneously by parameterizing $F$ with a neural network and performing JIT-compiled optimization using JAX and Optax.

## Features

- **Linear computational complexity**: $\mathcal{O}(N)$ scaling with respect to data points, scalable to large datasets.
- **Hardware accelerated**: Fast training via JAX JIT compilation on CPU/GPU/TPU.
- **Rational neural networks**: Built-in support for `RationalMLP` for smooth physical function approximation.
- **Multi-optimizer training**: Separate learning rates for neural network weights and physical critical exponents.
- **Uncertainty-aware fitting**: Negative log-likelihood (`NLLLoss`) weighting by Monte Carlo error bars.

## Installation

```bash
pip install jaxfss
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install jaxfss
```

## Quickstart

Check out the [Documentation and Tutorials](https://yonesuke.github.io/jaxfss/)!

## Related Packages

- [BSA (C++)](https://kenjiharada.github.io/BSA/): Finite-size scaling with Gaussian process regression.
- [FSS-tools (PyTorch)](https://github.com/KenjiHarada/FSS-tools): Finite-size scaling with neural network and Gaussian process.

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
