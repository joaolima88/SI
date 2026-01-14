import unittest
import numpy as np
from si.neural_networks.optimizers import Adam


class TestAdamOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up a clean optimizer instance before each test."""
        self.lr = 0.001
        self.optimizer = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    def test_initialization(self):
        """Test if the initial parameters are set correctly according to the specifications."""
        self.assertEqual(self.optimizer.learning_rate, 0.001)
        self.assertEqual(self.optimizer.beta_1, 0.9)
        self.assertEqual(self.optimizer.beta_2, 0.999)
        self.assertEqual(self.optimizer.epsilon, 1e-8)
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)

    def test_state_initialization_on_update(self):
        """Test if m and v are initialized with zeros upon the first update."""
        w = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.1, 0.1])

        self.optimizer.update(w, grad)

        # Check if they are no longer None
        self.assertIsNotNone(self.optimizer.m)
        self.assertIsNotNone(self.optimizer.v)

        # Check shapes
        self.assertEqual(self.optimizer.m.shape, w.shape)
        self.assertEqual(self.optimizer.v.shape, w.shape)

        # Check if t incremented
        self.assertEqual(self.optimizer.t, 1)

    def test_mathematical_correctness_step_1(self):
        """
        Manually verifies the mathematics of the first update step.

        Simplified Scenario:
        w = 1.0, grad = 0.1, lr = 0.1, b1 = 0.9, b2 = 0.999
        """
        optimizer = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        w = np.array([1.0])
        grad = np.array([0.1])

        # Execute update
        new_w = optimizer.update(w, grad)

        # --- Manual Calculation ---
        # t = 1
        # m = 0.9*0 + 0.1*0.1 = 0.01
        # v = 0.999*0 + 0.001*(0.1^2) = 0.00001
        # m_hat = 0.01 / (1 - 0.9) = 0.01 / 0.1 = 0.1
        # v_hat = 0.00001 / (1 - 0.999) = 0.00001 / 0.001 = 0.01
        # update = 0.1 * (0.1 / (sqrt(0.01) + 1e-8)) approx 0.1 * (0.1/0.1) = 0.1
        # w_expected = 1.0 - 0.1 = 0.9

        expected_w = 0.9

        # Use assertAlmostEqual due to floating point precision
        np.testing.assert_almost_equal(new_w[0], expected_w, decimal=5)

    def test_update_does_not_mutate_original_input(self):
        """Ensures the function returns new weights and does not mutate the input variable in-place."""
        w = np.array([1.0, 1.0])
        w_orig = w.copy()
        grad = np.array([0.1, 0.1])

        _ = self.optimizer.update(w, grad)

        np.testing.assert_array_equal(w, w_orig, err_msg="The original weights array should not be modified.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)