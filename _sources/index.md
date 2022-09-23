# `jaxfss` Reference Documentation
![](images/ising_binder.png)

`jaxfss` is a **finite-size scaling analysis** package.
It is built up on [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

## Finite-Size Scaling Analysis

$$
A(T,L)=L^{-c_{2}}F[(T-T_{\mathrm{c}})L^{c_{1}}]
$$

## Other packages
- Finite-size scaling package by Gaussian process with C++

    https://kenjiharada.github.io/BSA/
- Finite-size scaling package by neural network and Gaussian process with Python (PyTorch)

    https://github.com/KenjiHarada/FSS-tools

## Citation
Please cite this paper when you use this package for your research!!
- [Full paper] Ryosuke Yoneda and Kenji Harada, Neural Network Approach to Scaling Analysis of Critical Phenomena, [arXiv: 2209.01777](https://arxiv.org/abs/2209.01777).

    ```tex
    @article{yoneda2022neural,
        title={Neural Network Approach to Scaling Analysis of Critical Phenomena},
        author={Yoneda, Ryosuke and Harada, Kenji},
        url={https://arxiv.org/abs/2209.01777},
        journal={arXiv preprint arXiv:2209.01777},
        year={2022}
    }
    ```

- [Conference paper] Currently preparing!!