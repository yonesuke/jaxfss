# `jaxfss` Reference Documentation
![](images/ising_binder.png)

`jaxfss` is a **neural finite-size scaling analysis** package built on [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax).

## Finite-Size Scaling Analysis

The physical quantity near a critical point in a finite-size system obeys the scaling law:

$$
A(T, L) = L^{-c_2} F\left[ (T - T_{\mathrm{c}}) L^{c_1} \right]
$$

where $A(T, L)$ is a physical quantity at temperature $T$ in a finite-size system of size $L$. $T_{\mathrm{c}}$ is the critical temperature, and $c_1$ and $c_2$ are critical exponents.
Here $F[\cdot]$ is the unknown universal scaling function.

`jaxfss` estimates critical points and exponents by parameterizing the scaling function with a neural network and optimizing it end-to-end.

## Related Packages

- [BSA (C++)](https://kenjiharada.github.io/BSA/): Finite-size scaling with Gaussian processes.
- [FSS-tools (PyTorch)](https://github.com/KenjiHarada/FSS-tools): Finite-size scaling with neural networks and Gaussian processes.

## Citation

Please cite our paper published in *Physical Review E* when using `jaxfss`:

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