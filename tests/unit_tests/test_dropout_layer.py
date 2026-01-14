from unittest import TestCase
import numpy as np
from si.neural_networks.layers import Layer, Dropout


class TestDropoutLayerComplete(TestCase):

    def setUp(self):
        np.random.seed(123)
        self.input_data = np.random.rand(100, 100)
        self.dropout_rate = 0.4
        self.dropout_layer = Dropout(probability=self.dropout_rate)

    def test_class_inheritance(self):
        self.assertIsInstance(self.dropout_layer, Layer)

    def test_training_mode_behavior(self):
        np.random.seed(123)

        # Perform forward propagation in training mode
        output = self.dropout_layer.forward_propagation(self.input_data, training=True)

        # Verify output dimensions
        self.assertEqual(output.shape, self.input_data.shape)
        self.assertIsNotNone(self.dropout_layer.mask)

        # Calculate expected scaling factor
        scaling_factor = 1 / (1 - self.dropout_rate)
        dropout_mask = self.dropout_layer.mask.astype(bool)

        # 1. Verify deactivated neurons are strictly 0
        deactivated_units = output[~dropout_mask]
        self.assertTrue(np.all(deactivated_units == 0))

        # 2. Verify active neurons are properly scaled
        active_output = output[dropout_mask]
        active_input_scaled = self.input_data[dropout_mask] * scaling_factor

        # Compare with tolerance for floating point precision
        np.testing.assert_allclose(active_output, active_input_scaled, rtol=1e-7, atol=0)

    def test_inference_mode_integrity(self):
        output = self.dropout_layer.forward_propagation(self.input_data, training=False)
        np.testing.assert_array_equal(output, self.input_data)

    def test_backpropagation(self):
        np.random.seed(123)
        _ = self.dropout_layer.forward_propagation(self.input_data, training=True)
        dropout_mask = self.dropout_layer.mask.astype(bool)

        # Simulate gradient from next layer
        gradient_output = np.random.rand(*self.input_data.shape)

        # Calculate backward gradient
        gradient_input = self.dropout_layer.backward_propagation(gradient_output)

        self.assertEqual(gradient_input.shape, gradient_output.shape)

        # Gradient should be 0 where neurons were dropped
        self.assertTrue(np.all(gradient_input[~dropout_mask] == 0))

        # Gradient should pass through where neurons were kept
        np.testing.assert_array_equal(gradient_input[dropout_mask], gradient_output[dropout_mask])

    def test_output_shape(self):
        self.dropout_layer.forward_propagation(self.input_data, training=True)
        returned_shape = self.dropout_layer.output_shape()
        self.assertEqual(returned_shape, self.input_data.shape)

    def test_trainable_parameters_count(self):
        param_count = self.dropout_layer.parameters()
        self.assertEqual(param_count, 0)

    def test_wrong_initialization(self):
        with self.assertRaises(ValueError):
            Dropout(-1)
        with self.assertRaises(ValueError):
            Dropout(1)