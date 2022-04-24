from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self):
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.fitted_ = True

    def _fit(self, X: np.ndarray, y: np.ndarray, ) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        print(X.shape)
        self.rf = RandomForestClassifier(n_estimators=4)
        self.rf.fit(X, y)
        # Get numerical feature importances
        importances = list(self.rf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(list(X.columns), importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        # self.logisticRegr = LogisticRegression()
        # self.logisticRegr.fit(X, y)

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
        return self.rf.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : floatx
            Performance under loss function
        """
        return metrics.f1_score(self.predict(X), y, average='macro')
