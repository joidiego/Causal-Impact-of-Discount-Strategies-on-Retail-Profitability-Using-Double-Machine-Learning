# tests/test_double_ml.py
import unittest
import numpy as np
from src.models.double_ml import ContinuousDoubleML

class TestDoubleML(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 5)
        self.T = self.X[:, 0] + np.random.randn(200) * 0.5
        self.Y = 2.0 * self.T + self.X[:, 1] + np.random.randn(200)

    def test_fit_returns_effect(self):
        model = ContinuousDoubleML(n_folds=2, random_state=42)
        effect = model.fit(pd.DataFrame(self.X), self.T, self.Y)
        self.assertIsInstance(effect, float)
        self.assertGreater(effect, 1.5)
        self.assertLess(effect, 2.5)

if __name__ == '__main__':
    unittest.main()