import unittest
import numpy as np
from si.neural_networks.activation import SoftmaxActivation


class TestSoftmaxActivation(unittest.TestCase):

    def setUp(self):
        """Initializes layer instances before each test."""
        self.softmax = SoftmaxActivation()


    def test_softmax_probability_sum(self):
        """Ensures that the sum of Softmax outputs equals 1.0 (probability distribution)."""
        input_data = np.array([[1.0, 2.0, 3.0], [0.1, 0.5, 0.2]])
        output = self.softmax.activation_function(input_data)

        # Sum along axis 1 (rows)
        sums = np.sum(output, axis=1)

        # Verify if all sums are very close to 1
        np.testing.assert_array_almost_equal(sums, np.ones(sums.shape))

    def test_softmax_numerical_stability(self):
        """
        Critical Test: Checks if Softmax handles large numbers without generating NaNs,
        verifying the 'subtract max' stability trick.
        """
        # Without the max trick, exp(1000) would cause an overflow
        input_data = np.array([[1000.0, 1001.0, 1002.0]])
        output = self.softmax.activation_function(input_data)

        # There should be no NaNs (Not a Number)
        self.assertFalse(np.isnan(output).any())

        # Verify if the sum is still 1
        self.assertAlmostEqual(np.sum(output), 1.0)

        # Verify relative logic: 1002 is larger than 1001, so it should have a higher probability
        self.assertTrue(output[0, 2] > output[0, 1])

    def test_softmax_derivative(self):
        """Verifies the Softmax derivative implementation: f(x) * (1 - f(x))"""
        input_data = np.array([[1.0, 2.0, 3.0]])
        output = self.softmax.activation_function(input_data)  # f(x)
        derivative_output = self.softmax.derivative(input_data)

        # Expected calculation
        expected_derivative = output * (1 - output)

        np.testing.assert_array_almost_equal(derivative_output, expected_derivative)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)