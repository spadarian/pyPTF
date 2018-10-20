from pyPTF.datasets import load_water


def test_load_water():
    data = load_water()
    assert data.shape == (934, 5)
