import os
import jax.numpy as jnp
from flax import nnx
import optax
from softclip import SoftClip

import jaxfss

# Load critical data
data_path = os.path.join(os.path.dirname(__file__), "data", "ising-square-B.dat")
dataset = jaxfss.CriticalData.from_file(fname=data_path)
train_data = dataset.train_data
Ls, Ts, As = train_data["system_size"], train_data["temperature"], train_data["observable"]

# Define bijectors to enforce physical parameter bounds
bij_c1 = SoftClip(low=0.0)
bij_c2 = SoftClip(low=0.0)
bij_Tc = SoftClip(low=-1.0, high=1.0)


def bijector(params):
    p1, p2, pc = params
    return jnp.array([
        bij_c1.forward(p1),
        bij_c2.forward(p2),
        bij_Tc.forward(pc),
    ])


# Build NNX scaling function and unified FSS model
scaling_fn = jaxfss.RationalMLP(features=[20, 20, 1], din=1, rngs=nnx.Rngs(0))
model = jaxfss.FSSModel(scaling_fn=scaling_fn, n_critical=3, bijector=bijector)


# Define loss function
def loss_fn(m: jaxfss.FSSModel):
    y_pred = m(Ls, Ts)
    y_true = m.scaled_target(Ls, As)
    return jaxfss.MSELoss(y_true, y_pred)


# Optimize with separate learning rates for NN and critical exponents
optimizer = {
    "scaling_fn": optax.adam(learning_rate=1e-3),
    "fss": optax.adam(learning_rate=1e-2),
}
steps = 10000

# Learn!
losses, critical_vals = jaxfss.fit(model, loss_fn, optimizer, steps)

# Extract final physical parameters
c1, c2, scaled_Tc = model.critical_params
Tc = float(dataset.bij_temperature.inverse(scaled_Tc))
print(f"c1: {float(c1):.5f}, c2: {float(c2):.5f}, Tc: {Tc:.5f}")