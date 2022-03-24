from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt


pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()
    ug.fit(X)
    print(f"({ug.mu_}, {ug.var_})")
    # Question 2 - Empirically showing sample mean is consistent
    results = []
    for i in range(10, 1010, 10):
        results.append(abs(ug.mu_ - UnivariateGaussian.calculate_expectation(X[0:i])))
    plt.title("estimated expectation as number of samples increases")
    plt.xlabel("number of samples")
    plt.ylabel("distance between estimated & real expectation")
    plt.plot(np.arange(10, 1010, 10), results, color="red")
    plt.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    plt.title("samples to PDFs scatter plot")
    plt.xlabel("sample value")
    plt.ylabel("pdf value")
    plt.scatter(X, ug.pdf(X), color="red")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X2 = np.random.multivariate_normal(mu, sigma, 1000)
    mg = MultivariateGaussian()
    mg.fit(X2)
    print(mg.mu_)
    print(mg.cov_)
    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    values = np.zeros(shape=(f1.size, f3.size))
    for i in range(f1.size):
        for j in range(f3.size):
            mu = np.array([f1[i], 0, f3[j], 0])
            values[i][j] = MultivariateGaussian.log_likelihood(mu, sigma, X2)
    plt.title("log likelihood for each [f1, 0, f3, 0] model")
    plt.xlabel("f1 values")
    plt.ylabel("f3 values")
    heatmap = plt.pcolor(values)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('log likelihood')
    plt.xticks(np.arange(0, 200, 9.7), np.arange(-10, 11))
    plt.yticks(np.arange(0, 200, 9.7), np.arange(-10, 11))
    fig = plt.gcf()
    fig.set_dpi(700)
    plt.show()
    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(values), values.shape)
    # rounding to 4 because 3 produces a float precision error.
    print(np.around(f1[ind[0]], 4), np.around(f3[ind[1]], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
