import unittest
import numpy as np
from si.neural_networks.activation import TanhActivation

class TestTanhActivation(unittest.TestCase):

    def setUp(self):
        """Initializes layer instances before each test."""
        self.tanh = TanhActivation()

    def test_tanh_activation_values(self):
        """Verifies if Tanh computes values correctly based on known inputs."""
        input_data = np.array([[0.0], [10.0], [-10.0]])
        output = self.tanh.activation_function(input_data)

        # Tanh(0) should be 0
        self.assertAlmostEqual(output[0, 0], 0.0)
        # Tanh(large positive number) should be close to 1
        self.assertAlmostEqual(output[1, 0], 1.0)
        # Tanh(large negative number) should be close to -1
        self.assertAlmostEqual(output[2, 0], -1.0)

    def test_tanh_derivative(self):
        """Verifies the Tanh derivative formula: 1 - tanh(x)^2"""
        input_data = np.array([[0.0], [0.5]])
        computed_derivative = self.tanh.derivative(input_data)

        # Derivative of tanh at 0 is 1 - 0^2 = 1
        self.assertAlmostEqual(computed_derivative[0, 0], 1.0)

        # Manual verification for 0.5
        expected_val = 1 - np.tanh(0.5) ** 2
        self.assertAlmostEqual(computed_derivative[1, 0], expected_val)



if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)