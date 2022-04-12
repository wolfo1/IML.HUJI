import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    # create day of year column
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # delete invalid samples (temp is way too low for the specific cities)
    df = df[df['Temp'] > -50]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == 'Israel']
    fig, ax = plt.subplots()
    grouped = df_israel.groupby('Year')
    for name, group in grouped:
        ax.plot(group.DayOfYear, group.Temp, marker='o', linestyle='', ms=1, label=name)
    plt.title("Temperature by Day of Year, color coded by year")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature")
    ax.legend(loc=2, prop={'size': 5})
    plt.show()
    # plot 2: Month to standard deviation of temp
    grouped = df_israel.groupby('Month').agg('std')
    plt.bar(x=grouped.index, height=grouped.Temp)
    plt.title("Standard deviation of daily Temperature by Month")
    plt.xlabel("Month")
    plt.ylabel("std (temp)")
    plt.show()
    # Question 3 - Exploring differences between countries
    fig, ax = plt.subplots()
    for country in df['Country'].unique():
        df_country = df[df['Country'] == country]
        std_group = df_country.groupby(['Country', 'Month']).agg('std')
        avg_group = df_country.groupby(['Country', 'Month']).agg('mean')
        ax.errorbar(np.arange(1, 13),
                    avg_group.Temp,
                    std_group.Temp,
                    capsize=4,
                    elinewidth=1,
                    markeredgewidth=1, label=country)
    plt.title("Avg temperature by Month & Country, with Error Bars")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature")
    plt.legend()
    plt.show()
    # Question 4 - Fitting model for different values of `k`
    # df_israel = df_israel.sample(frac=1)  # shuffle all rows randomly
    X = df_israel['DayOfYear']
    y = df_israel['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)
    results = []
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_X.to_numpy(), train_y.to_numpy())
        error = round(polyfit.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        results.append(error)
        print(f"k = {k}, MSE = {error}")
    plt.bar(np.arange(10), results)
    plt.xticks(np.arange(10), np.arange(1, 11))
    plt.title("Prediction Loss as a function of poly degree k")
    plt.xlabel("k value")
    plt.ylabel("MSE")
    plt.show()
    # Question 5 - Evaluating fitted model on different countries
    polyfit = PolynomialFitting(5)
    polyfit.fit(df_israel['DayOfYear'].to_numpy(), df_israel['Temp'].to_numpy())
    results = []
    countries = list(df['Country'].unique())
    countries.remove('Israel')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for country in countries:
        df_country = df[df['Country'] == country]
        X = df_country['DayOfYear'].to_numpy()
        y = df_country['Temp'].to_numpy()
        results.append(polyfit.loss(X, y))
    fig, ax = plt.subplots()
    bars = ax.bar(countries, results, color=colors)
    ax.bar_label(bars)
    plt.ylabel("MSE")
    plt.title("MSE by country from data learned on Israel")
    plt.show()
