from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    positive_features = ['price', 'floors', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15']
    nonegative_features = ['sqft_basement', 'yr_renovated']
    for feature in positive_features:
        df = df[df[feature] > 0]
    for feature in nonegative_features:
        df = df[df[feature] >= 0]
    zip_groups = df.groupby('zipcode').agg({'price': 'mean'})
    zip_groups['zipcode_group'] = pd.cut(zip_groups['price'], bins=7, labels=['0',
                                                                              '1',
                                                                              '2',
                                                                              '3',
                                                                              '4',
                                                                              '5',
                                                                              '6'],
                                         include_lowest=True)
    zip_groups = zip_groups.drop(columns="price")
    df = pd.merge(df, zip_groups,
                  left_on='zipcode',
                  how='left',
                  right_index=True)
    df = pd.get_dummies(df, columns=['zipcode_group'],
                        drop_first=True)
    price = df['price']
    df['date'] = pd.to_datetime(df.date, format='%Y%m%dT%H%M%S')
    df['year'] = df['date'].dt.year
    df['yr_renovated_built'] = df[['yr_built', 'yr_renovated']].max(axis=1)
    df = df.drop(columns=['id', 'date', 'price', 'yr_built', 'yr_renovated', 'zipcode'])
    return df, price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for x in X:
        cov = np.cov(X[x], y)
        sig = np.std(X[x]) * np.std(y)
        if sig != 0:
            pearson = (cov / sig)[0, 1]
        else:
            pearson = np.array(0)
        plt.scatter(X[x], y)
        plt.title(f"Plot Feature {x} and price.\n Pearson corr = {pearson.round(5)}")
        plt.xlabel(x)
        plt.ylabel("price")
        save_file = os.path.join(output_path, f"plot_corr_feature_{x}.png")
        plt.savefig(save_file)
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, response = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, response, ".\plots")
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, response, 0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # Generate the x values of the test set
    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        test_loss = []
        for i in range(10):
            X_p = train_X.sample(frac=p / 100)
            y_p = train_y[X_p.index]
            lr = LinearRegression()
            lr.fit(X_p.to_numpy(), y_p.to_numpy())
            test_loss.append(lr.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_loss.append(np.mean(test_loss))
        std_loss.append(np.std(test_loss))
    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)
    fig, ax = plt.subplots()
    ax.plot(np.arange(10, 101), mean_loss)
    ax.fill_between(np.arange(10, 101), (mean_loss - 2 * std_loss),
                    (mean_loss + 2 * std_loss), color='b', alpha=.1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title("Loss as a function of % of data learned")
    plt.ylabel("Mean loss of 10 iterations")
    plt.xlabel("%p of train data learned")
    plt.show()
