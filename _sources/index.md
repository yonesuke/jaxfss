# `jaxfss` Reference Documentation
![](images/ising_binder.png)

`jaxfss` is a **finite-size scaling analysis** package.
It is built up on [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

## Finite-Size Scaling Analysis
The physical quantity near a critical point in a finite-size system obeys the scaling law written as

$$
A(T,L)=L^{-c_{2}}F[(T-T_{\mathrm{c}})L^{c_{1}}]
$$

where $A(T, L)$ is a physical quantity at temperature $T$ in a finite-size system of which size is $L$. $T_{\mathrm{c}}$ is a critical temperature. The exponent $c_1$ and $c_2$ are critical exponents.
Here $F[\cdot]$ is a scaling function. Unfortunately, we do not know the scaling function’s form in advance. Thus, we need to infer not only the value of critical temperature and exponents but also the scaling function itself from given data.

`jaxfss` is a package for those who need to analyze critical phenomena and calculate the critical point and critical exponents from the finite system size data.
The basic idea is that the scaling function is well approximated by some neural network function.
It is made up of JAX and Flax, and you can easily use.
The idea of this package is so simple that you can extend it to your need if it is not sufficient for you.

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