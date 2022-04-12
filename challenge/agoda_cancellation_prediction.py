from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_policy(policy: str, stay_days: int):
    if policy == 'UNKNOWN':
        return 0, 100
    policies = policy.split("_")
    first_policy = policies[0]
    d_idx = first_policy.find('D')
    if first_policy[-1] == 'N':
        days = int(first_policy[d_idx + 1: -1])
        P = str(int(round((days / stay_days) * 100, 0)))
    else:
        P = first_policy[d_idx+1: -1]
    return int(first_policy[:d_idx]), P

def load_data(train_filename: str, test_filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    train_filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(train_filename).drop_duplicates()
    test_data = pd.read_csv(test_filename).drop_duplicates()
    test_size = test_data.shape[0]
    full_data = pd.concat([full_data, test_data])
    y = full_data['cancellation_datetime'].fillna(0)
    y[y != 0] = 1
    y = y.astype(int)
    df = pd.DataFrame(full_data, columns=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_country_code',
                                          'hotel_star_rating', 'accommadation_type_name', 'charge_option',
                                          'h_customer_id', 'customer_nationality', 'guest_is_not_the_customer',
                                          'guest_nationality_country_name', 'no_of_adults', 'no_of_children',
                                          'no_of_extra_bed', 'no_of_room', 'origin_country_code',
                                          'original_selling_amount', 'original_payment_method',
                                          'original_payment_type', 'original_payment_currency',
                                          'is_user_logged_in', 'is_first_booking',
                                          'request_nonesmoke', 'request_latecheckin', 'request_highfloor',
                                          'request_largebed', 'request_twinbeds', 'request_airport',
                                          'request_earlycheckin', 'cancellation_policy_code'])
    df = pd.get_dummies(data=df, columns=['hotel_country_code', 'accommadation_type_name', 'charge_option',
                                          'customer_nationality', 'guest_nationality_country_name',
                                          'origin_country_code', 'original_payment_method',
                                          'original_payment_type', 'original_payment_currency'])
    # create stay_days & booking_to_checkin days columns. change columns to day of year.
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    df['booking_to_checkin'] = (df['booking_datetime'] - df['checkin_date']).dt.days
    df['stay_days'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df["checkin_date"] = pd.to_datetime(df["checkin_date"]).dt.dayofyear
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"]).dt.dayofyear
    df.drop('checkout_date', axis=1, inplace=True)
    # parse police code
    df['policy'] = df.apply(lambda x: parse_policy(policy=x['cancellation_policy_code'], stay_days=x['stay_days']), axis=1)
    df[['cancel_days', 'cancel_fine']] = pd.DataFrame(df.policy.tolist(), index=df.index)
    df.drop('policy', axis=1, inplace=True)
    df.drop('cancellation_policy_code', axis=1, inplace=True)
    # change true/false to 0/1
    df["is_user_logged_in"] = df["is_user_logged_in"].astype(int)
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    df = df.fillna(0)
    end_idx = df.shape[0]
    test_X = df[end_idx - test_size:]
    test_y = y[end_idx - test_size:]
    train_X = df[:end_idx - test_size]
    train_y = y[:end_idx - test_size]
    return train_X, train_y, test_X, test_y


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
    # Load data
    train_X, train_y, test_X, test_y = load_data("../datasets/agoda_cancellation_train.csv", "test_set_week_2.csv")
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    print(estimator.loss(test_X, test_y))
    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "204867881_316563949_207090119.csv")
