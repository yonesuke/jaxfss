# Neural Scaling Analysis (NSA)

Finite-size scaling (FSS) analysis is a fundamental technique in statistical physics and computational condensed matter physics for investigating critical phenomena and phase transitions from finite-size numerical simulation data.

**Neural Scaling Analysis (NSA)** is a machine learning framework that formulates finite-size scaling analysis as an end-to-end regression problem using neural networks. This package `jaxfss` provides a fast, JIT-compiled, flexible implementation of NSA built on top of [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax).

---

## 1. Finite-Size Scaling Hypothesis

Near a continuous phase transition at a critical point $T_{\mathrm{c}}$, correlation lengths diverge and physical observables obey universal scaling relations. In a finite-size system with linear dimension $L$ at temperature $T$, an observable $A(T, L)$ follows the finite-size scaling law:

$$
A(T, L) = L^{-c_2} F\left[ (T - T_{\mathrm{c}}) L^{c_1} \right]
$$

where:
* $T_{\mathrm{c}}$ is the critical point (e.g., critical temperature).
* $c_1 = 1/\nu$ is the scaling exponent related to the correlation length exponent $\nu$.
* $c_2$ is the anomalous dimension or scaling exponent of observable $A$.
* $F[\cdot]$ is the **universal scaling function**, which is generally unknown a priori.

The goal of scaling analysis is to estimate the critical parameters $(T_{\mathrm{c}}, c_1, c_2)$ such that data points $(X, Y) = \left( (T - T_{\mathrm{c}}) L^{c_1}, A(T, L) L^{c_2} \right)$ from different system sizes $L$ collapse onto a single master curve $F[X]$.

---

## 2. Motivation: Why Neural Networks?

Traditionally, two main methods have been used to perform finite-size scaling analysis:

1. **Polynomial regression**:
   * Approximates the scaling function $F[X]$ with a low-order Taylor polynomial.
   * *Limitations*: Only valid in a narrow region around the critical point ($X \approx 0$). Prone to overfitting or underfitting if the polynomial degree is mischosen.
2. **Gaussian Process (GP) regression** (e.g., Harada's Bayesian Scaling Analysis):
   * Provides non-parametric, flexible curve fitting with error estimation.
   * *Limitations*: Computational complexity scales as $\mathcal{O}(N^3)$ where $N$ is the number of data points, making it prohibitively heavy for large-scale or dense Monte Carlo datasets.

**Neural Scaling Analysis (NSA)** overcomes both limitations:
* **High expressiveness**: Universal approximation capability of neural networks models arbitrary smooth scaling functions across wide parameter windows.
* **Linear computational complexity**: With modern automatic differentiation and mini-batch / full-batch gradient descent in JAX, the computational cost scales as $\mathcal{O}(N)$.
* **GPU/TPU acceleration**: NSA easily scales to massive data using hardware acceleration.

---

## 3. Method Architecture

In NSA, the scaling function $F$ is parameterized by a neural network $\mathsf{NN}_{\theta}$:

$$
Y = \mathsf{NN}_{\theta}(X)
$$

The network parameters $\theta$ and the critical parameters $\phi = (c_1, c_2, T_{\mathrm{c}})$ are jointly optimized by minimizing a loss function:

$$
\min_{\theta, \phi} \mathcal{L}(\theta, \phi)
$$

### Data Pre-Processing (`CriticalData`)
Neural networks train best when inputs and targets are standardized to order $\mathcal{O}(1)$. `CriticalData` automatically applies affine transformations:
* Rescales system size $L \in (0, 1]$.
* Rescales temperature $T \in [-1, 1]$.
* Rescales observable $A$.

### Parameter Constraints (Bijectors)
Physical exponents often have known constraints (e.g., $c_1 > 0$, or $T_{\mathrm{c}} \in [T_{\min}, T_{\max}]$).
Using bijectors (such as `softclip.SoftClip`), unconstrained optimization variables $p$ are mapped to physical parameters:
* $c_1 = \mathrm{SoftClip}(p_1; \mathrm{low}=0.0)$
* $T_{\mathrm{c}} = \mathrm{SoftClip}(p_c; \mathrm{low}=T_{\min}, \mathrm{high}=T_{\max})$

### Activation Functions (`RationalMLP`)
Standard ReLU or sigmoid networks can have derivative discontinuities or saturation issues. `RationalMLP` uses trainable rational functions $P(x)/Q(x)$ as activations, offering superior smoothness and expressiveness for physical functions.

---

## 4. Loss Functions

### Mean Squared Error (`MSELoss`)
When observational uncertainties are uniform or negligible:

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^N \left( Y_i - \mathsf{NN}_{\theta}(X_i) \right)^2
$$

### Negative Log-Likelihood Loss (`NLLLoss`)
When Monte Carlo errors $\sigma_i$ are provided for each data point:

$$
\mathcal{L}_{\mathrm{NLL}} = \frac{1}{2N} \sum_{i=1}^N \left[ \frac{(Y_i - \mathsf{NN}_{\theta}(X_i))^2}{\sigma_{Y, i}^2} + \log(2\pi \sigma_{Y, i}^2) \right]
$$

where the scaled variance is $\sigma_{Y, i}^2 = \sigma_i^2 L_i^{2 c_2}$.

---

## 5. Dual Optimization Strategy

Because the neural network weights $\theta$ and critical parameters $\phi$ have different optimization landscapes, it is advantageous to use separate learning rates:
* $\mathrm{lr}_{\mathrm{mlp}} \sim 10^{-3}$ for neural network parameters
* $\mathrm{lr}_{\mathrm{fss}} \sim 10^{-2}$ for critical exponents and critical point

`jaxfss.fit` supports passing a dictionary of Optax optimizers:

```python
optimizer = {
    "mlp": optax.adam(learning_rate=1e-3),
    "fss": optax.adam(learning_rate=1e-2),
}
params, losses, critical_vals = jaxfss.fit(loss_fn, optimizer, init_params, steps=10000)
```

---

## Reference

If you use NSA or `jaxfss` in your research, please cite:
* R. Yoneda and K. Harada, *Neural network approach to scaling analysis of critical phenomena*, **Phys. Rev. E 107, 044128 (2023)**. [doi:10.1103/PhysRevE.107.044128](https://doi.org/10.1103/PhysRevE.107.044128).
