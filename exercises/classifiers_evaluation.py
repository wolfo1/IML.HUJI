from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(
            callback=lambda perceptron, data, labels: losses.append(perceptron.loss(data[:, 1:], labels)))
        perceptron.fit(X, y)
        # Plot figure of loss as function of fitting iteration
        fig = px.line(losses, title=n + 'data, loss by iteration', width=1000, height=600)
        fig.update_layout(xaxis_title='iteration', yaxis_title='loss', title_x=0.5)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_pred = gnb.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc = accuracy(y, lda_pred)
        gnb_acc = accuracy(y, gnb_pred)
        fig = make_subplots(1, 2, subplot_titles=("GNB with " + str(np.round(gnb_acc, 5)) + " accuracy",
                                                  "LDA with " + str(np.round(lda_acc, 5)) + " accuracy"))
        fig.update_layout(title_text="Dataset: " + f, title_x=0.5,
                          width=1000,
                          height=600)
        # Add traces for data-points setting symbols and colors
        symbols = np.array(['cross', 'diamond', 'star'])
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        for i, model in enumerate([gnb, lda]):
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                                     marker=dict(color=y, symbol=symbols[y],
                                                 colorscale=['red', 'yellow', 'blue'],
                                                 line=dict(color='black', width=1))),
                          row=1, col=i + 1)
            fig.add_trace(decision_surface(model.predict, lims[0], lims[1], showscale=False),
                          row=1, col=i + 1)
        # Add `X` dots specifying fitted Gaussians' means
        for i, model in enumerate([gnb, lda]):
            fig.add_trace(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode='markers',
                                     marker=dict(color='red', symbol='x'), showlegend=False),
                          row=1, col=i + 1)
        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, model in enumerate([gnb, lda]):
            for g in range(3):
                fig.add_trace(get_ellipse(model.mu_[g], model.cov_ if i == 1 else np.diag(model.vars_[g])),
                              row=1, col=i + 1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
