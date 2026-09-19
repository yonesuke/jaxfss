import os
import jax.numpy as jnp
import numpy as np
import pytest
from jaxfss import CriticalData


@pytest.fixture
def sample_data():
    Ls = jnp.array([16.0, 32.0, 64.0, 16.0, 32.0, 64.0])
    Ts = jnp.array([2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
    As = jnp.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    As_err = jnp.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    return Ls, Ts, As, As_err


def test_critical_data_init(sample_data):
    Ls, Ts, As, As_err = sample_data
    dataset = CriticalData(Ls, Ts, As, As_err)

    assert dataset.n_data == 6
    assert dataset.Ls.shape == (6, 1)
    assert dataset.Ts.shape == (6, 1)
    assert dataset.As.shape == (6, 1)
    assert dataset.As_err.shape == (6, 1)

    assert dataset.maximum_system_size == 64.0
    assert float(dataset.system_size.max()) == pytest.approx(1.0)
    assert "system_size" in dataset.train_data
    assert "temperature" in dataset.train_data
    assert "observable" in dataset.train_data
    assert "observable_var" in dataset.train_data

    # Test backward-compatible alias
    assert dataset.training_data is dataset.train_data
    assert "jaxfss.CriticalData" in repr(dataset)


def test_critical_data_from_file():
    filepath = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "ising.txt")
    if os.path.exists(filepath):
        dataset = CriticalData.from_file(filepath)
        assert dataset.n_data > 0
        assert dataset.maximum_system_size > 0
