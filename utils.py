import torch

from typing import List

import algos


def make_logistic_regression(N: int, d: int, alpha: float, seed: int):
    """
    N: number of datapoints
    d: number of features
    alpha: regularisation parameter
    seed: parameter for reproducibility
    """
    torch.manual_seed(seed=seed)

    X = torch.rand(N, d) * 4 - 2
    X /= d**0.5
    w_star = torch.ones(d)
    y = torch.sign(X @ w_star + torch.randn(N))

    # Replace any exact zeros (unlikely but just in case) with +1
    y[y == 0] = 1.0

    def f(w: torch.Tensor) -> torch.Tensor:
        """
        L2-regularised logistic regression loss.

        f(w) = (1/N) sum_i log(1 + exp(-y_i * x_i^T w)) + (alpha/2) ||w||^2
        """
        logits = X @ w
        margins = y * logits
        # numerically stable log(1 + exp(-m))
        loss = torch.nn.functional.softplus(-margins).mean()
        reg = 0.5 * alpha * torch.sum(torch.square(w))
        return loss + reg

    return f, X, y


def make_linear_regression(N: int, d: int, alpha: float, seed: int):
    """
    N: number of datapoints
    d: number of features
    alpha: regularisation parameter
    seed: parameter for reproducibility
    """
    torch.manual_seed(seed=seed)

    X = torch.rand(N, d) * 4 - 2
    X /= d**0.5
    w_star = torch.ones(d)
    y = X @ w_star + torch.randn(N)

    def f(w: torch.Tensor) -> torch.Tensor:
        """
        L2-regularised linear regression loss.

        f(w) = 1/(2N)||y - Xw||^{2} + (alpha/2) ||w||^2
        """
        return 0.5 * torch.mean(torch.square(y - X @ w)) + 0.5 * alpha * torch.sum(
            torch.square(w)
        )

    return f, X, y


def find_best_eta(
    etas: List[float],
    alg: algos.Algorithm,
    total_iters: int,
    alpha: float,
    x_0: torch.Tensor,
    **kwargs,
):
    best_eta = None
    best_f = float("inf")
    for eta in etas:
        alg.set_eta(eta)
        if isinstance(alg, algos.AGD):
            momentum = None
            if alpha > 0:
                # strongly convex
                momentum = (1 - (alpha * eta) ** 0.5) / (1 + (alpha * eta) ** 0.5)
            _, f_vals = alg.run(x_0=x_0, K=total_iters, momentum=momentum)
        elif isinstance(alg, algos.DHFA):
            N, K = algos.make_NK_for_dhfa(
                eta=eta,
                total_iters=total_iters,
                alpha=alpha,
                lam=kwargs.get("lam", None),
            )
            _, f_vals = alg.run(x_0=x_0, K=K, N=N)
        else:
            _, f_vals = alg.run(x_0=x_0, K=total_iters)
        if min(f_vals) < best_f:
            best_eta = eta
            best_f = min(f_vals)
    print(f"Found best step size for {alg.__class__.__name__}: {best_eta:4f}")
    return best_eta, best_f
