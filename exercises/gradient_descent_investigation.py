import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    # TODO: needs model?
    def callback(model, **kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])
        return

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        print("eta:", eta)
        l1 = L1(init.copy())
        recorder = get_gd_state_recorder_callback()
        model = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=recorder[0])
        model_fit = model.fit(l1, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L1, np.array(recorder[2]), title=f"L1. eta={eta}"))
        plt.plot(list(range(len(recorder[1]))), recorder[1], )
        plt.xlabel("iters")
        plt.ylabel("norm")
        plt.title(f"l1 norm as function of number of iterations. eta={eta}")
        plt.grid()
        plt.show()
        l1.weights = model_fit
        print("L1 module with lowest error:", l1.compute_output())

        l2 = L2(init.copy())
        recorder = get_gd_state_recorder_callback()
        model = GradientDescent(learning_rate=FixedLR(eta), out_type="best", callback=recorder[0])
        model_fit = model.fit(l2, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L2, np.array(recorder[2]), title=f" L2 module with eta={eta}"))
        plt.plot(list(range(len(recorder[1]))), recorder[1], )
        plt.title(f"l2 norm as function of number of iterations. eta={eta}")
        plt.xlabel("iters")
        plt.ylabel("norm")
        plt.grid()
        plt.show()
        l2.weights = model_fit
        print("L2 module with lowest error:", l2.compute_output())


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    conv = []

    for gamma in gammas:
        recorder = get_gd_state_recorder_callback()
        model = GradientDescent(ExponentialLR(eta, gamma), 1e-5, 1000, 'best', recorder[0])
        l1 = L1(init.copy())
        model.fit(l1, X=None, y=None)
        conv.append(recorder[1])
    # Plot algorithm's convergence for the different values of gamma
    for i in range(len(gammas)):
        plt.plot(list(range(len(conv[i]))), conv[i])
    plt.title('l1 norm as iterations')
    plt.xlabel('iters')
    plt.ylabel('norm')
    plt.legend(gammas)
    plt.grid()
    plt.show()

    print("L1 min norm:", np.min([np.min(conv[i]) for i in range(4)]))

    # Plot descent path for gamma=0.95
    gamma = 0.95
    recorder = get_gd_state_recorder_callback()
    model = GradientDescent(ExponentialLR(eta, gamma), 1e-5, 1000, 'best', recorder[0])
    l1 = L1(init.copy())
    model.fit(l1, X=None, y=None)
    decay = recorder[2]
    plotly.offline.plot(plot_descent_path(L1, np.array(decay)))
    plotly.offline.plot(plot_descent_path(L2, np.array(decay)))


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    from sklearn.metrics import roc_curve, auc
    from utils import custom
    f, t, threshold = roc_curve(y_train.to_numpy(), model.predict_proba(np.c_[np.ones(len(X_train.to_numpy())), X_train.to_numpy()]))
    go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', dash='dash')),
                    go.Scatter(x=f, y=t, mode='markers+lines', text=threshold, showlegend=False,
                               marker_color=[custom[0], custom[-1]],
                               hovertemplate='<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}')],
              layout=go.Layout(title=rf'$\text{{ROC Curve}}={auc(f, t):.6f}$',
                               xaxis=dict(title=r'$\text{False Positive Rate }$'),
                               yaxis=dict(title=r'$\text{True Positive Rate }$'))).show()
    print('alpha: ', round(threshold[np.argmax(t - f)], 2), 'loss: ', model.loss(X_test.to_numpy(), y_test.to_numpy()))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection.cross_validate import cross_validate
    from IMLearn.metrics.loss_functions import misclassification_error as mse
    lam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ['l1', 'l2']:
        train_error = []
        val_error = []
        model = LogisticRegression(penalty=penalty, alpha=0.5,
                                   solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))
        for reg_param in lam:
            model.lam_ = reg_param
            t, v = cross_validate(model, X_train.to_numpy(), y_train.to_numpy(), mse)
            train_error.append(t)
            val_error.append(v)
        model.lam_ = lam[np.argmin(val_error)]
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        print('model:', penalty, 'error:', model.loss(X_test.to_numpy(), y_test.to_numpy()), 'lambda=', model.lam_)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
