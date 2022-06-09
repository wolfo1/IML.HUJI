from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
import agoda_process_data


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)
    week_to_test = 8
    k = 150
    train_X, train_y, test_X, test_y = agoda_process_data.load_data("../datasets/agoda_cancellation_train.csv",
                                                                    f"test_weeks_data/week_{week_to_test}_test_data.csv",
                                                                    range(1, week_to_test))
    estimator = AgodaCancellationEstimator(k)
    estimator.fit(train_X, train_y)
    evaluate_and_export(estimator, test_X, "204867881_316563949_207090119.csv")
    # print f1 score if labels exist
    # try:
    #     true_y = pd.read_csv(f"test_weeks_labels/week_{week_to_test}_labels.csv")['cancel']
    #     print(estimator.loss(test_X, true_y))
    # except IOError:
    #     pass
