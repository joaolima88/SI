import os
import numpy as np
import unittest
from unittest import TestCase
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        self.cce = CategoricalCrossEntropy()

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

################ 14

    def test_loss_correctness_simple(self):
        """
        Verifies if the loss calculation is mathematically correct
        in a simple scenario without zeros.
        Formula: -sum(y_true * log(y_pred))
        """
        # Example: 1 sample, 3 classes. Class 0 is the true one.
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])

        # Expected calculation: -1 * log(0.7) - 0 - 0 = 0.35667...
        expected_loss = -np.log(0.7)

        loss = self.cce.loss(y_true, y_pred)
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_loss_numerical_stability(self):
        """
        Verifies if np.clip is working to avoid log(0).
        Without clip, this would result in -inf.
        """
        y_true = np.array([[1, 0]])
        # Exact '0' prediction would cause math error without clip
        y_pred = np.array([[0.0, 1.0]])

        loss = self.cce.loss(y_true, y_pred)

        # The value should be finite (approx -log(1e-15) ≈ 34.53)
        self.assertFalse(np.isinf(loss), "Loss should not be infinite (clip failure?)")
        self.assertFalse(np.isnan(loss), "Loss should not be NaN")
        self.assertGreater(loss, 0)

    def test_derivative_shapes_and_values(self):
        """
        Verifies if the derivative returns the correct shape and expected values.
        Formula: -y_true / y_pred
        """
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.5, 0.25, 0.25]])

        # Expected Derivative:
        # Class 1: -1 / 0.5 = -2.0
        # Class 2: -0 / 0.25 = 0
        # Class 3: -0 / 0.25 = 0
        expected_grad = np.array([[-2.0, -0.0, -0.0]])

        grad = self.cce.derivative(y_true, y_pred)

        # Check dimensions
        self.assertEqual(grad.shape, y_true.shape)
        # Check values
        np.testing.assert_array_almost_equal(grad, expected_grad)

    def test_derivative_stability(self):
        """
        Verifies if the derivative handles division by zero using clip.
        """
        y_true = np.array([[1, 0]])
        y_pred = np.array([[0.0, 1.0]])  # '0.0' would cause division by zero

        grad = self.cce.derivative(y_true, y_pred)

        # The gradient at index 0 should be very large (negative), but finite
        self.assertFalse(np.isinf(grad).any())
        self.assertFalse(np.isnan(grad).any())

        # Since y_pred is clipped to 1e-15, the gradient should be -1/1e-15 = -1e15
        self.assertTrue(grad[0, 0] < -1e10)

    def test_batch_processing(self):
        """
        Verifies if it works for a data batch (multiple rows).
        """
        # 2 samples, 2 classes
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])

        # Expected loss: -(log(0.9) + log(0.8))
        expected_loss = -(np.log(0.9) + np.log(0.8))

        loss = self.cce.loss(y_true, y_pred)
        self.assertAlmostEqual(loss, expected_loss, places=5)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)