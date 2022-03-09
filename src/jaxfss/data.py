import numpy as np
import jax.numpy as jnp
import distrax

class CriticalData:
    """
    Data handler of critical data
    """
    def __init__(self, Ls, Ts, As, As_err):
        self._Ls = Ls
        self._Ts = Ts
        self._As = As
        self._As_err = As_err

        self.n_data = len(self._Ls)

        # reshape
        self.Ls = self._Ls.reshape(-1, 1)
        self.Ts = self._Ts.reshape(-1, 1)
        self.As = self._As.reshape(-1, 1)
        self.As_err = self._As_err.reshape(-1, 1)

        # scaling system size
        self.maximum_system_size = self.Ls.max()
        self.bij_system_size = distrax.ScalarAffine(
            shift=0.0,
            scale=1.0/self.maximum_system_size
        )
        self.system_size = self.bij_system_size.forward(self.Ls)

        # scaling temperature
        max_idx = self.Ls == self.maximum_system_size
        temp_min, temp_max = self.Ts[max_idx].min(), self.Ts[max_idx].max()
        self.bij_temperature = distrax.ScalarAffine(
            shift = -(temp_max + temp_min)/(temp_max-temp_min),
            scale = 2.0/(temp_max-temp_min)
        )
        self.temperature = self.bij_temperature.forward(self.Ts)

        # scaling observable
        obs_min, obs_max = self.As.min(), self.As.max()
        self.bij_observable = distrax.ScalarAffine(
            shift=0.0,
            scale=1.0/(obs_max-obs_min)
        )
        self.observable = self.bij_observable.forward(self.As)
        self.observable_err = self.bij_observable.forward(self.As_err)
        self.observable_var = self.observable_err ** 2

        # training data
        self.train_data = {
            "system_size": self.system_size,
            "temperature": self.temperature,
            "observable": self.observable,
            "observable_var": self.observable_var
        }

    @classmethod
    def from_file(cls, fname):
        Ls, Ts, As, As_err = jnp.array(np.loadtxt(fname, unpack=True))
        return cls(Ls, Ts, As, As_err)

    def __repr__(self):
        return f"jaxfss.CriticalData(n_data={self.n_data})"