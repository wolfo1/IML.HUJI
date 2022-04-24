from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # get all labels. find MU and PI
        self.classes_ = np.unique(y)
        self.mu_ = []
        for i in self.classes_:
            self.mu_.append(np.mean(X[y == i], axis=0))
        self.mu_ = np.array(self.mu_)
        self.pi_ = np.array([np.mean(y == i) for i in self.classes_])
        # build COV matrix based on recitation formula
        for i in range(len(self.classes_)):
            x_i = X[y == self.classes_[i]] - self.mu_[i, :]
            if self.cov_ is None:
                self.cov_ = np.matmul(np.transpose(x_i), x_i)
            else:
                self.cov_ += np.matmul(np.transpose(x_i), x_i)
        self.cov_ = (1 / y.size) * self.cov_
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # multiply each column i by pi[i]. return argmax class.
        max_matrix = np.multiply(self.likelihood(X), self.pi_)
        return self.classes_[max_matrix.argmax(1)]
        # mu_tranposed = np.transpose(self.mu_)
        # X_cov_mu = np.matmul(np.transpose(self._cov_inv), mu_tranposed)
        # return np.argmax(np.matmul(X, X_cov_mu) + -0.5 * np.diag(np.matmul(np.matmul(self.mu_, self._cov_inv), mu_tranposed)) + np.log(self.pi_), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood_matrix = []
        det_cov = det(self.cov_)

        def pdf(mu, x_i):
            half_1 = np.sqrt((2 * np.pi) ** x_i.size * det_cov)
            half_2 = -0.5 * np.matmul((x_i - mu), np.matmul(self._cov_inv, (x_i - mu)))
            return 1 / half_1 * np.exp(half_2)

        for i in range(self.classes_.size):
            class_likelihood = np.apply_along_axis(lambda x_i: pdf(self.mu_[i], x_i), arr=X, axis=1)
            likelihood_matrix.append(class_likelihood)
        return np.array(likelihood_matrix).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
