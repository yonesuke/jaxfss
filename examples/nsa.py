import jaxfss
import jax
import jax.numpy as jnp
import optax
from softclip import SoftClip

# initializing MLP
mlp = jaxfss.RationalMLP(features=[20, 20, 1])
mlp_params = mlp.init(jax.random.PRNGKey(0), jnp.array([[1]]))

# creating data
dataset = jaxfss.CriticalData.from_file(fname="data/ising-square-B.dat")
train_data = dataset.train_data

# creating bijector for stablizing learning
bij_c1 = SoftClip(low=0.0)
bij_c2 = SoftClip(low=0.0)
bij_Tc = SoftClip(low=-1.0, high=1.0)

# initial parameters
init_params = {
    "mlp": mlp_params,
    "fss": jnp.zeros(3)
}

# helper function for bijectors
def bijectors(params):
    p1, p2, pc = params["fss"]
    c1 = bij_c1.forward(p1)
    c2 = bij_c2.forward(p2)
    scaled_Tc = bij_Tc.forward(pc)
    return jnp.array([c1, c2, scaled_Tc])

# loss function for learning
def loss_fn(params):
    c1, c2, scaled_Tc = bijectors(params)
    Ls, Ts, As = train_data["system_size"], train_data["temperature"], train_data["observable"]
    X = (Ts - scaled_Tc) * Ls ** c1
    Y = As * Ls ** c2
    return jaxfss.MSELoss(mlp.apply(params["mlp"], X), Y)

lr = 10**-3
optimizer = optax.adam(learning_rate=lr)
steps = 10**4

# learn!!
params, losses, fsses = jaxfss.fit(loss_fn, optimizer, init_params, steps)
c1, c2, scaled_Tc = bijectors(params)
Tc = dataset.bij_temperature.inverse(scaled_Tc)
print(c1, c2, Tc)