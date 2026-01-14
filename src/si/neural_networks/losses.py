from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)


###### 14

class CategoricalCrossEntropy(LossFunction):
    """
    Implements the categorical cross-entropy loss function,
    commonly used for multi-class classification tasks.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded true labels.
        y_pred : np.ndarray
            Predicted probabilities for each class.

        Returns
        -------
        float
            The computed categorical cross-entropy loss.
        """
        # Ensure numerical stability by clipping predictions
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Compute the cross-entropy loss
        return -np.sum(y_true * np.log(clipped_predictions))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded true labels.
        y_pred : np.ndarray
            Predicted probabilities for each class.

        Returns
        -------
        np.ndarray
            The gradient of the loss with respect to the predictions.
        """
        # Ensure numerical stability by clipping predictions
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Compute the gradient
        return -y_true / clipped_predictions
