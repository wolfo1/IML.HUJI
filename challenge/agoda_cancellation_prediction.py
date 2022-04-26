from sklearn.preprocessing import MinMaxScaler

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
from sklearn.ensemble import *
from sklearn import metrics
# conversions from USD to all currencies
conversions = {'USD': 1, 'AED': 3.6725, 'AFN': 87.5007, 'ALL': 111.4952, 'AMD': 467.558, 'ANG': 1.79, 'AOA': 404.0973, 'ARS': 114.04, 'AUD': 1.3726, 'AWG': 1.79, 'AZN': 1.6979, 'BAM': 1.8108, 'BBD': 2.0, 'BDT': 85.438, 'BGN': 1.8106, 'BHD': 0.376, 'BIF': 2024.8096, 'BMD': 1.0, 'BND': 1.3669, 'BOB': 6.8653, 'BRL': 4.6806, 'BSD': 1.0, 'BTN': 76.4959, 'BWP': 11.973, 'BYN': 2.8555, 'BZD': 2.0, 'CAD': 1.2669, 'CDF': 1999.8809, 'CHF': 0.9563, 'CLP': 822.3729, 'CNY': 6.5225, 'COP': 3728.9547, 'CRC': 657.6806, 'CUP': 24.0, 'CVE': 102.0895, 'CZK': 22.4939, 'DJF': 177.721, 'DKK': 6.9072, 'DOP': 54.912, 'DZD': 143.832, 'EGP': 18.5802, 'ERN': 15.0, 'ETB': 51.3614, 'EUR': 0.9259, 'FJD': 2.1163, 'FKP': 0.7787, 'FOK': 6.9072, 'GBP': 0.7788, 'GEL': 3.0339, 'GGP': 0.7787, 'GHS': 7.7553, 'GIP': 0.7787, 'GMD': 54.0333, 'GNF': 8896.9671, 'GTQ': 7.6475, 'GYD': 209.0387, 'HKD': 7.8479, 'HNL': 24.5693, 'HRK': 6.9759, 'HTG': 107.894, 'HUF': 343.295, 'IDR': 14341.489, 'ILS': 3.2735, 'IMP': 0.7787, 'INR': 76.4723, 'IQD': 1458.072, 'IRR': 42051.3384, 'ISK': 128.7315, 'JEP': 0.7787, 'JMD': 154.654, 'JOD': 0.709, 'JPY': 128.7001, 'KES': 115.7729, 'KGS': 82.9306, 'KHR': 4041.021, 'KID': 1.3734, 'KMF': 455.4913, 'KRW': 1241.5203, 'KWD': 0.2996, 'KYD': 0.8333, 'KZT': 443.6302, 'LAK': 13106.4208, 'LBP': 1507.5, 'LKR': 330.8464, 'LRD': 152.0024, 'LSL': 15.5633, 'LYD': 4.712, 'MAD': 9.6981, 'MDL': 18.4927, 'MGA': 3991.8343, 'MKD': 56.6224, 'MMK': 1835.3117, 'MNT': 3052.3832, 'MOP': 8.0833, 'MRU': 36.4208, 'MUR': 42.6761, 'MVR': 15.4107, 'MWK': 819.5117, 'MXN': 20.2706, 'MYR': 4.3037, 'MZN': 64.6108, 'NAD': 15.5633, 'NGN': 414.9575, 'NIO': 35.8503, 'NOK': 8.9409, 'NPR': 122.3934, 'NZD': 1.5043, 'OMR': 0.3845, 'PAB': 1.0, 'PEN': 3.7455, 'PGK': 3.5245, 'PHP': 52.3739, 'PKR': 186.6637, 'PLN': 4.2895, 'PYG': 6827.8499, 'QAR': 3.64, 'RON': 4.5623, 'RSD': 108.8545, 'RUB': 77.0753, 'RWF': 1051.2487, 'SAR': 3.75, 'SBD': 7.9427, 'SCR': 14.4082, 'SDG': 445.0241, 'SEK': 9.5371, 'SGD': 1.3669, 'SHP': 0.7787, 'SLL': 12368.3272, 'SOS': 577.9904, 'SRD': 20.7337, 'SSP': 425.1448, 'STN': 22.6835, 'SYP': 2517.89, 'SZL': 15.5633, 'THB': 34.0252, 'TJS': 12.4745, 'TMT': 3.4991, 'TND': 2.819, 'TOP': 2.2329, 'TRY': 14.7711, 'TTD': 6.7809, 'TVD': 1.3734, 'TWD': 29.2194, 'TZS': 2316.5256, 'UAH': 29.523, 'UGX': 3522.2721, 'UYU': 40.3923, 'UZS': 11347.4483, 'VES': 4.4354, 'VND': 22974.0933, 'VUV': 111.8606, 'WST': 2.5658, 'XAF': 607.3217, 'XCD': 2.7, 'XDR': 0.7358, 'XOF': 607.3217, 'XPF': 110.4843, 'YER': 250.3169, 'ZAR': 15.5636, 'ZMW': 17.0195, 'ZWL': 153.7166}


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
    if stay_days < 0:
        return 100
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
    df = pd.read_csv(train_filename).drop_duplicates()
    df['cancellation_datetime'] = df['cancellation_datetime'].fillna(0)
    df['cancellation_datetime'] = pd.to_datetime(df['cancellation_datetime'])
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    df['new_datetime'] = (df['cancellation_datetime'] - df['booking_datetime']).dt.days
    df['new_datetime'][df['new_datetime'] > 45] = 0
    df['new_datetime'][df['new_datetime'] < 7] = 0
    df['another_datetime'] = (df['checkin_date'] - df['cancellation_datetime']).dt.days
    df['new_datetime'][df['another_datetime'] < 2] = 0
    df['new_datetime'][df['new_datetime'] != 0] = 1
    df['cancellation_datetime'] = df['new_datetime']
    df.drop(['new_datetime', 'another_datetime'], axis=1, inplace=True)
    # more_data = pd.read_csv('test_set_week_1.csv').drop_duplicates()
    # df = pd.concat([df, more_data])
    # more_data = pd.read_csv('test_set_week_2.csv').drop_duplicates()
    df = pd.concat([df, more_data])
    test_data = pd.read_csv(test_filename).drop_duplicates()
    test_size = test_data.shape[0]
    full_data = pd.concat([df, test_data])
    df = pd.DataFrame(full_data, columns=['booking_datetime', 'checkin_date', 'checkout_date',
                                          'hotel_star_rating',
                                          # 'accommadation_type_name',
                                          'charge_option',
                                          'customer_nationality',
                                          'guest_is_not_the_customer',
                                          # 'guest_nationality_country_name',
                                          'no_of_adults', 'no_of_children',
                                          # 'no_of_extra_bed'
                                          'no_of_room',
                                          'original_selling_amount',
                                          # 'original_payment_method',
                                          # 'original_payment_type',
                                          'original_payment_currency',
                                          'is_user_logged_in', 'is_first_booking',
                                          # 'request_nonesmoke', 'request_latecheckin', 'request_highfloor',
                                          # 'request_largebed', 'request_twinbeds', 'request_airport',
                                          'request_earlycheckin',
                                          'cancellation_policy_code', 'cancellation_datetime',
                                          # 'hotel_city_code'
                                          # 'hotel_chain_code',
                                          # 'hotel_brand_code',
                                          'hotel_area_code',
                                          # 'hotel_country_code'
                                          ])
    df = df[df['original_selling_amount'] < 20000]
    df = pd.get_dummies(data=df, columns=[
        # 'accommadation_type_name',
        'charge_option',
        'customer_nationality',
        # 'guest_nationality_country_name',
        # 'original_payment_method',
        # 'original_payment_type'
        #'original_payment_currency'
    ], drop_first=True)
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    # groups
    groups_labels = ['hotel_area_code']
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
    df['booking_to_checkin'] = (df['booking_datetime'] - df['checkin_date']).dt.days
    df['stay_days'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df["checkin_date"] = pd.to_datetime(df["checkin_date"]).dt.dayofyear
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"]).dt.dayofyear
    # parse police code
    df['policy'] = df.apply(lambda x: parse_policy2(policy=x['cancellation_policy_code'], stay_days=x['stay_days']),
                            axis=1)
    # convert currency sums to USD
    # df['original_selling_amount'] = df.apply(lambda x: (x['original_selling_amount'] * 1 / (conversions[x['original_payment_currency']])), axis=1)
    # change true/false to 0/1
    df["is_user_logged_in"] = df["is_user_logged_in"].astype(int)
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    y = df['cancellation_datetime']
    df.drop(['cancellation_policy_code', 'original_payment_currency', 'cancellation_datetime', 'checkout_date'], axis=1, inplace=True)
    df = df.fillna(0)
    sc = MinMaxScaler()
    df = sc.fit_transform(df)
    end_idx = df.shape[0]
    test_X = df[end_idx - test_size:]
    test_y = y[end_idx - test_size:]
    train_X = df[:end_idx - test_size]
    train_y = y[:end_idx - test_size]
    return train_X, train_y, test_X, test_y
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
    # test on week1, week2
    # for k in ["test_set_week_1.csv", "test_set_week_2.csv"]:
    #     train_X, train_y, test_X, test_y = load_data("../datasets/agoda_cancellation_train.csv", k)
    #     scores = []
    #     for i in range(5):
    #         lr = RandomForestClassifier(n_estimators=80, class_weight={0: 1, 1: 3})
    #         lr.fit(train_X, train_y)
    #         scores.append(metrics.f1_score(test_y, lr.predict(test_X), average='macro'))
    #     print(np.average(scores))

    train_X, train_y, test_X, test_y = load_data("../datasets/agoda_cancellation_train.csv", "test_set_week_3.csv")
    lr = RandomForestClassifier(n_estimators=80)
    lr.fit(train_X, train_y)
    evaluate_and_export(lr, test_X, "204867881_316563949_207090119.csv")
    y = df = pd.read_csv('test_set_week_3_labels.csv')
    y_pred = pd.read_csv('204867881_316563949_207090119.csv')
    print(y)
    print(y_pred)
    print(metrics.f1_score(y, y_pred, average='macro'))
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
