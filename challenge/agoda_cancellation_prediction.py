from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
import agoda_process_data
from sklearn.metrics import f1_score


def evaluate_and_export(result: np.ndarray, filename: str):
    pd.DataFrame(result, columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)
    week_to_test = 10
    test_X = pd.read_csv(f"test_weeks_data/week_{week_to_test}_test_data.csv")['h_booking_id']
    result = np.zeros(test_X.shape)
    # merge the results of multiple models
    for k in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45,  50, 100, 150]:
        train_X, train_y, test_X, test_y = agoda_process_data.load_data("../datasets/agoda_cancellation_train.csv",
                                                                        f"test_weeks_data/week_{week_to_test}_test_data.csv",
                                                                        range(1, week_to_test))
        estimator = AgodaCancellationEstimator(k)
        estimator.fit(train_X, train_y)
        k_res = estimator.predict(test_X)
        result = np.maximum(k_res, result)
    # export results to CSV
    evaluate_and_export(result, "204867881_316563949_207090119.csv")
    # print f1 score if labels exist
    # try:
    #     test_y = pd.read_csv("204867881_316563949_207090119.csv")["predicted_values"]
    #     true_y = pd.read_csv(f"test_weeks_labels/week_{week_to_test}_labels.csv")['cancel']
    #     print(f"week {week_to_test}. score:", f1_score(test_y, true_y, average='macro'))
    # except IOError:
    #     print("done!")
