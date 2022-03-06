import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize
from scipy.stats import multivariate_normal
from collections import OrderedDict


################################################################################
## Datasets
################################################################################


def generate_binary(n, dist0, dist1, bias=True):
    X = np.vstack((dist0.rvs(n), dist1.rvs(n)))
    if bias:
        X = np.hstack((np.ones((2 * n, 1)), X))
    y = np.concatenate([np.zeros(n, dtype=np.int), np.ones(n, dtype=np.int)])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


# two-dimensional with bias term, binary classification, separable
np.random.seed(1)
X1, y1 = generate_binary(
    100,
    multivariate_normal([3, 4], np.identity(2)),
    multivariate_normal([-2, 7], np.identity(2)),
)

# two-dimensional with bias term, binary classification, not separable
np.random.seed(1)
X2, y2 = generate_binary(
    100,
    multivariate_normal([3, 4], np.identity(2)),
    multivariate_normal([-1, 5], np.identity(2)),
)

# one-dimensional, regression
np.random.seed(1)
X3 = np.linspace(0, 4 * np.pi, 100).reshape((100, 1))
y3 = np.sin(X3) + np.random.normal(0, 0.1, X3.shape[0])[:, np.newaxis]
X3test = np.linspace(0, 4 * np.pi, 123).reshape((123, 1))
y3test = np.sin(X3test) + np.random.normal(0, 0.1, X3test.shape[0])[:, np.newaxis]

# store in pytorch tensors (call .numpy() to get back to numpy)
X3 = torch.from_numpy(X3).float()
y3 = torch.from_numpy(y3).float()
X3test = torch.from_numpy(X3test).float()
y3test = torch.from_numpy(y3test).float()


################################################################################
## Plotting
################################################################################


def abline(slope, intercept, color=None, label=None):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = intercept + slope * x
    axes.set_autoscale_on(False)
    plt.plot(x, y, color=color, label=label)
    axes.set_autoscale_on(True)


def plot2(X, y):
    """Scatter plot 2D points (X) with binary labels (y)"""
    plt.scatter(X[y == 0, 1], X[y == 0, 2], c="red", label="negative")
    plt.scatter(X[y == 1, 1], X[y == 1, 2], c="green", label="positive")
    plt.legend()


def plot2db(w, color=None, label=None):
    """Add the given decision boundary into the current plot"""
    intercept = -w[0] / w[2]
    slope = -w[1] / w[2]
    abline(slope, intercept, color, label)
    if label is not None:
        plt.legend()


def plot2dbs(X, y, n=10, maxepochs=100, pocket=False):
    """Scatter plot 2D points (X) with binary labels (y), the decision boundaries of
n runs of the pocket algorithm, of SVM, and of logistic regression."""
    N, D = X.shape
    plot2(X, y)

    mrperceptron = N
    for i in range(n):
        w = pt_train(X, y, maxepochs=maxepochs, pocket=pocket, w0=torch.rand(3))
        label = "pocket" if pocket else "perceptron"
        plot2db(w, color="lightgray", label=label if i == 0 else None)
        mrperceptron = min(mrperceptron, torch.sum(torch.abs(pt_classify(X, w) - y)))

    w = svm.LinearSVC(fit_intercept=False).fit(X, y).coef_[0]
    w = torch.from_numpy(w).float()
    plot2db(w, label="SVM")
    mrsvm = torch.sum(torch.abs(pt_classify(X, w) - y))

    w = LogisticRegression(fit_intercept=False).fit(X, y).coef_[0]
    w = torch.from_numpy(w).float()
    plot2db(w, label="LogReg")
    mrlogreg = torch.sum(torch.abs(pt_classify(X, w) - y))

    print()
    print("Misclassification rates (train)")
    print("Perceptron (best result): {:d}".format(int(mrperceptron)))
    print("Linear SVM (C=1)        : {:d}".format(int(mrsvm)))
    print("Logistic regression     : {:d}".format(int(mrlogreg)))


def plot1(X, y, label=None):
    """Scatter plot 1D points (X) with real label (y)"""
    if type(X) is torch.Tensor:
        X = X.numpy()
    if type(y) is torch.Tensor:
        y = y.numpy()

    plt.plot(X, y, linestyle=" ", marker="x", label=label)
    plt.xlabel("x")
    plt.ylabel("y")


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def plot1fit(X, model, label="fit", hidden=False, scale=True, alpha=0.3):
    """Scatter plot of (x,y) points + fit of a pytorch model"""
    lines = plt.plot(X, model(X).detach().numpy(), label=label)

    if hidden:
        ax = plt.gca()
        if scale:
            ax2 = plt.twinx()
        else:
            ax2 = plt.twinx()
        plt.sca(ax)

        W2 = model.state_dict()["output.weight"]
        b2 = model.state_dict()["output.bias"]
        W1 = model.state_dict()["linear1.weight"]
        b1 = model.state_dict()["linear1.bias"]
        transfer = model._modules["nonlin1"]
        h = transfer(X @ W1.t() + b1).float()
        for i in range(h.shape[1]):
            label = "$h_" + str(i) + "$"
            if scale:
                lines += ax2.plot(
                    X,
                    (h[:, i] * W2[0, i]).numpy(),
                    label=label + " scaled",
                    alpha=alpha,
                )
            else:
                lines += ax2.plot(X, (h[:, i]).numpy(), label=label, alpha=alpha)

    plt.legend(lines, [l.get_label() for l in lines])


################################################################################
## PyTorch training
################################################################################


def fnn_model(sizes, transfer=lambda: nn.Sigmoid()):
    """Construct a PyTorch FNN.

    The FNN is fully-connected between subsequent layers. sizes contains the layer sizes
    (input size, hidden sizes, output size) and the specified non-linearity is used for
    all hidden layers. The output layer is linear.

    """
    layers = OrderedDict()
    for i in range(len(sizes) - 2):  # hidden layers
        l = nn.Linear(sizes[i], sizes[i + 1])
        torch.nn.init.xavier_normal_(l.weight)
        layers["linear" + str(i + 1)] = l
        layers["nonlin" + str(i + 1)] = transfer()
    layers["output"] = nn.Linear(sizes[-2], sizes[-1])
    torch.nn.init.xavier_normal_(layers["output"].weight)
    return nn.Sequential(layers)


def fnn_train(
    X,
    y,
    model,
    optimizer=None,
    loss=torch.nn.MSELoss(),
    max_epochs=2000,
    tol=1e-6,
    verbose=False,
):
    """Train the given PyTorch model.

    Supports SGD-based optimizers and L-BFGS (default).

    """
    if type(model) is list:
        model = fnn_model(model)
    if optimizer is None:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, history_size=1000)

    cost = math.inf

    def eval():
        nonlocal cost
        yhat = model(X)
        cost = loss(yhat, y)
        cost_item = cost.item()
        if verbose:
            print(f"{cost: 8.3f}", end=" ")
        optimizer.zero_grad()
        cost.backward()
        return cost

    for epoch in range(max_epochs):
        last_cost = cost
        if verbose:
            print("Epoch {: 5d}: loss=".format(epoch), end="")
        if type(optimizer) is torch.optim.LBFGS:
            optimizer.step(eval)
        else:
            eval()
            optimizer.step()
        if verbose:
            print()
        if (abs(cost - last_cost) < tol) or cost > 2 * last_cost or math.isnan(cost):
            # converged or blow-up?
            break

    return model


################################################################################
## scipy training
################################################################################

## train via scikit learn
def pack_parameters(model, gradients=False):
    """Pack all parameters of a PyTorch model into a numpy array.

    If gradients is set, pack gradients instead.

    """
    numel = sum(map(lambda result: result.numel(), model.parameters()))
    result = np.ndarray(numel)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        if gradients:
            result[offset : (offset + n)] = p.grad.view(-1).detach()
        else:
            result[offset : (offset + n)] = p.data.view(-1).detach()
        offset += n
    return result


def unpack_parameters(packed_pars, model):
    "Unpack parameters of a PyTorch model previously packed with pack_parameters."
    numel = len(packed_pars)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data[...] = torch.FloatTensor(packed_pars[offset : (offset + n)]).view(
            p.shape
        )
        offset += n


def eval_model(packed_pars, model, X, y, loss):
    "Evaluate a PyTorch model using packed parameters. Returns loss + gradients."
    unpack_parameters(packed_pars, model)
    yhat = model(X)
    cost = loss(yhat, y)
    cost.backward()
    grad = pack_parameters(model, True)
    for p in model.parameters():
        p.grad.data.zero_()
    return cost.item(), grad


def train_scipy(
    X,
    y,
    model,
    loss=torch.nn.MSELoss(),
    method="BFGS",
    options={"gtol": 1e-6, "disp": True, "maxiter": 10000},
):
    "Train a PyTorch model using scipy.optimize."
    # double precision needed for BFGS
    Xdouble = X.double()
    ydouble = y.double()
    for p in model.parameters():
        p.data = p.data.double()

    # evaluate the model
    scipy.optimize.minimize(
        eval_model,
        pack_parameters(model),
        args=(model, Xdouble, ydouble, loss),
        jac=True,
        method=method,
        options=options,
    )

    # back to flots
    for p in model.parameters():
        p.data = p.data.float()

    # done
    return model
