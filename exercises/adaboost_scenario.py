import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(lambda: DecisionStump(), n_learners)
    ada.fit(train_X, train_y)
    print("noise", noise, "done fitting")
    train_loss = []
    test_loss = []
    for i in range(1, n_learners + 1):
        train_loss.append(ada.partial_loss(train_X, train_y, i))
        test_loss.append(ada.partial_loss(test_X, test_y, i))
    fig = go.Figure([
        go.Scatter(x=np.arange(1, n_learners + 1), y=train_loss, name='Loss over train data'),
        go.Scatter(x=np.arange(1, n_learners + 1), y=test_loss, name='Loss over test data')])
    fig.update_layout(
        title=f"Q1: Loss as a function of n_learners over train & test Data.\nnoise={noise}.",
        xaxis=dict(title="n_learners"),
        yaxis=dict(title="Loss"))
    fig.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(['star', 'cross'])
    train_y_sym = np.ones(train_y.shape[0]).astype(int)
    test_y_sym = np.zeros(test_y.shape[0]).astype(int)
    symbols_y = np.concatenate((train_y_sym, test_y_sym), axis=0)
    fig = make_subplots(2, 2, subplot_titles=(str(T[0]) + " learners",
                                              str(T[1]) + " learners",
                                              str(T[2]) + " learners",
                                              str(T[3]) + " learners"),
                        horizontal_spacing=0.05, vertical_spacing=0.05)
    fig.update_layout(title=f"Q2: Decision boundary as function of number of learners\nnoise={noise}.")
    X = np.concatenate([train_X, test_X], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    for i, iters in enumerate(T):
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=y,
                                             symbol=symbols[symbols_y],
                                             colorscale=['red', 'blue'],
                                             line=dict(color='black', width=1),
                                             size=4)),
                      row=i // 2 + 1, col=i % 2 + 1)
        fig.add_trace(decision_surface(lambda X: ada.partial_predict(X, iters), lims[0], lims[1], showscale=False),
                      row=i // 2 + 1, col=i % 2 + 1)
    # make it prettier
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best = np.argmin(test_loss)
    acc = accuracy(test_y, ada.partial_predict(test_X, best))
    fig = make_subplots()
    fig.update_layout(
        title=f"Q3: Decision surface of best performing ensemble, {best} learners with accuracy {acc}.\nnoise={noise}.")
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                             marker=dict(color=test_y, colorscale=['red', 'blue'], line=dict(color='black', width=1),
                                         size=4)))
    fig.add_trace(decision_surface(lambda X: ada.partial_predict(X, iters), lims[0], lims[1], showscale=False))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    # Question 4: Decision surface with weighted samples
    fig = make_subplots()
    D = ada.D_ / np.max(ada.D_) * 5
    fig.update_layout(title=f"Q4: Weighted decision surface.\nnoise={noise}.")
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                             marker=dict(color=train_y, colorscale=['red', 'blue'], line=dict(color='black', width=1),
                                         size=D)))
    fig.add_trace(decision_surface(lambda X: ada.partial_predict(X, iters), lims[0], lims[1], showscale=False))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)
