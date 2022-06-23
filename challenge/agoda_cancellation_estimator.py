from __future__ import annotations
from typing import NoReturn
import numpy as np
# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def rfc_best_params(train_X, train_y):
    param_grid = {
        'n_estimators': [2, 45, 80, 100, 150, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'Balanced', 'balanced_subsample']
    }
    rfc = RandomForestClassifier()
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='f1_macro')
    CV_rfc.fit(train_X, train_y)
    print(CV_rfc.best_params_)


class AgodaCancellationEstimator:
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, estimators):
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        self.rfc = RandomForestClassifier(n_estimators=estimators, class_weight='balanced')

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
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

        self.rfc.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        return self.rfc.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
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
