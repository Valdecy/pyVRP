import sys
import os
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub missing external dependencies
fake_folium = types.ModuleType("folium")
fake_folium.plugins = types.ModuleType("plugins")
sys.modules.setdefault("folium", fake_folium)
sys.modules.setdefault("folium.plugins", fake_folium.plugins)

fake_pandas = types.ModuleType("pandas")
sys.modules.setdefault("pandas", fake_pandas)

fake_matplotlib = types.ModuleType("matplotlib")
fake_pyplot = types.ModuleType("pyplot")
class FakeStyle:
    def use(self, *args, **kwargs):
        pass
fake_pyplot.style = FakeStyle()
fake_matplotlib.pyplot = fake_pyplot
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)

class FakeArray(list):
    @property
    def T(self):
        return [list(t) for t in zip(*self)]

fake_numpy = types.ModuleType("numpy")

def array(seq):
    return FakeArray(seq)

def cumsum(seq):
    out = []
    total = 0
    for val in seq:
        total += val
        out.append(total)
    return out

fake_numpy.array = array
fake_numpy.cumsum = cumsum
sys.modules.setdefault("numpy", fake_numpy)

from src.pyVRP import evaluate_distance

class SimpleMatrix:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return [self.data[r][c] for r, c in zip(rows, cols)]
        return self.data[key]

def test_evaluate_distance_simple():
    matrix = SimpleMatrix([
        [0, 1, 3],
        [1, 0, 2],
        [3, 2, 0],
    ])
    result = evaluate_distance(matrix, [0], [1, 2])
    assert result == [0.0, 1, 3, 6]
