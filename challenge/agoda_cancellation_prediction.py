from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn import metrics
from sklearn.preprocessing import *
from sklearn import tree

def days_to_P(policy, days):
    d_idx = policy.find('D')
    fine_days = int(policy[d_idx + 1: -1])
    if policy[-1] == 'N':
        P = str(int(round((fine_days / days) * 100, 0)))
    else:
        P = str(int(policy[d_idx + 1: -1]))
    fine_days = policy[:d_idx]
    return fine_days, P


def parse_policy2(policy, stay_days):
    if policy == 'UNKNOWN':
        return 100
    default = "100"
    policies = policy.split("_")
    if len(policies) == 1:
        days, P = days_to_P(policies[0], stay_days)
        return int(days + P)
    if len(policies) == 2:
        if 'D' not in policies[1]:
            # no second step
            day1, P1 = days_to_P(policies[0], stay_days)
            day2, P2 = days_to_P(policies[1], stay_days)
            return int(day1 + P1 + P2)
        else:
            day1, P1 = days_to_P(policies[0], stay_days)
            day2, P2 = days_to_P(policies[1], stay_days)
            return int(day1 + P1 + day2 + P2)
    else:
        day1, P1 = days_to_P(policies[0], stay_days)
        day2, P2 = days_to_P(policies[1], stay_days)
        day3, P3 = days_to_P(policies[2], stay_days)
        return int(day1 + P1 + day2 + P2 + P3)


def load_data(train_filename: str, test_filename=None):
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
    full_data = pd.read_csv(train_filename).drop_duplicates()
    test_data = pd.read_csv(test_filename).drop_duplicates()
    test_data['cancellation_datetime'] = 0
    test_size = test_data.shape[0]
    full_data = pd.concat([full_data, test_data])
    df = pd.DataFrame(full_data, columns=['booking_datetime', 'checkin_date', 'checkout_date',
                                          'hotel_star_rating',
                                          # 'accommadation_type_name',
                                          'charge_option',
                                          # 'customer_nationality',
                                          'guest_is_not_the_customer',
                                          # 'guest_nationality_country_name',
                                          'no_of_adults', 'no_of_children',
                                          # 'no_of_extra_bed'
                                          'no_of_room',
                                          'original_selling_amount',
                                          # 'original_payment_method',
                                          # 'original_payment_type',
                                          # 'original_payment_currency',
                                          'is_user_logged_in', 'is_first_booking',
                                          # 'request_nonesmoke', 'request_latecheckin', 'request_highfloor',
                                          # 'request_largebed', 'request_twinbeds', 'request_airport',
                                          # 'request_earlycheckin',
                                          'cancellation_policy_code', 'cancellation_datetime',
                                          'hotel_city_code', 'hotel_chain_code',
                                          # 'hotel_brand_code',
                                          'hotel_area_code',
                                          'hotel_country_code'])
    df = df[df['original_selling_amount'] < 20000]
    df = pd.get_dummies(data=df, columns=[
        # 'accommadation_type_name'
        'charge_option',
        # 'customer_nationality',
        # 'guest_nationality_country_name',
        # 'original_payment_method',
        # 'original_payment_type'
        # 'original_payment_currency'
    ], drop_first=True)
    df['cancellation_datetime'] = df['cancellation_datetime'].fillna(0)
    df['cancellation_datetime'][df['cancellation_datetime'] != 0] = 1
    df['cancellation_datetime'] = df['cancellation_datetime'].astype(int)
    groups_labels = ['hotel_country_code', 'hotel_city_code', 'hotel_chain_code', 'hotel_area_code']
    for x in groups_labels:
        groups = df.groupby(x).agg({'cancellation_datetime': 'mean'})
        groups[x + '_group'] = pd.cut(groups['cancellation_datetime'], bins=7, include_lowest=True)
        groups = groups.drop(columns="cancellation_datetime")
        df = pd.merge(df, groups,
                      left_on=x,
                      how='left',
                      right_index=True)
        df = pd.get_dummies(df, columns=[x + '_group'],
                            drop_first=True)
        df.drop(x, axis=1, inplace=True)
    # create stay_days & booking_to_checkin days columns. change columns to day of year.
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    df['booking_to_checkin'] = (df['booking_datetime'] - df['checkin_date']).dt.days
    df['stay_days'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df = df[df['stay_days'] > 0]
    df["checkin_date"] = pd.to_datetime(df["checkin_date"]).dt.dayofyear
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"]).dt.dayofyear
    y = df['cancellation_datetime']
    df.drop('cancellation_datetime', axis=1, inplace=True)
    df.drop('checkout_date', axis=1, inplace=True)
    # parse police code
    df['policy'] = df.apply(lambda x: parse_policy2(policy=x['cancellation_policy_code'], stay_days=x['stay_days']),
                            axis=1)
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
    # sc = MinMaxScaler()
    # df = sc.fit_transform(df)
    # return df, y


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
    # Load data
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data("../datasets/agoda_cancellation_train.csv", "test_set_week_1.csv")
    lr = RandomForestClassifier(n_estimators=2)
    lr.fit(train_X, train_y)
    evaluate_and_export(lr, test_X, "204867881_316563949_207090119.csv")
    # df, labels = load_data("../datasets/agoda_cancellation_train.csv")
    # for j in [4, 10, 15, 25, 30, 45]:
    #     f1_scores = []
    #     for i in range(3):
    #         train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=65 / 100, random_state=0)
    #         # random forest
    #         lr = RandomForestClassifier(n_estimators=j)
    #         lr.fit(train_X, train_y)
    #         test_X = test_X.sample(n=700)
    #         test_y = test_y[test_X.index]
    #         y_pred = lr.predict(test_X)
    #         f1 = metrics.f1_score(test_y, y_pred, average='macro')
    #         f1_scores.append(f1)
    #     print("rf f1 avg: ", j, np.average(f1_scores))
    # Fit model over data
    # print(estimator.loss(test_X, test_y))
    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "204867881_316563949_207090119.csv")
