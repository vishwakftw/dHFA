import numpy as np
import torch

from argparse import ArgumentParser
from matplotlib import pyplot as plt

import algos
import utils

torch.set_default_dtype(torch.float64)

plt.rcParams.update({"text.usetex": True, "font.size": 22, "axes.grid": True})

p = ArgumentParser()
p.add_argument(
    "--N",
    type=int,
    help="Number of samples",
)
p.add_argument("--d", type=int, help="Number of dimensions")
p.add_argument("--beta", type=float, help="Regularisation parameter")
p.add_argument("--seed", type=int, default=1729, help="Seed for reproducibility")
p.add_argument("--total_iters", type=int, help="Number of iterations")
p.add_argument("--lam", type=float, default=(1 + 3**0.5) / 2, help="Lambda parameter")
args = p.parse_args()

x_0 = torch.zeros(args.d)
f, X, y = utils.make_linear_regression(
    N=args.N, d=args.d, alpha=args.beta, seed=args.seed
)
eigvals_XTX = torch.linalg.eigvalsh(X.T @ X).clamp_min_(torch.tensor(0.0))
L, m = eigvals_XTX.max() / args.N + args.beta, eigvals_XTX.min() / args.N + args.beta

if m > 0:
    momentum = (1 - (m / L) ** 0.5) / (1 + (m / L) ** 0.5)
    lam = 0
else:
    momentum = None
    lam = args.lam

gd = algos.GD(f=f, eta=1 / L)
x_gd_sol, f_vals_gd = gd.run(x_0=x_0, K=args.total_iters)

agd = algos.AGD(f=f, eta=1 / L)
x_agd_sol, f_vals_agd = agd.run(
    x_0=x_0,
    K=args.total_iters,
    momentum=momentum,
)

dhfaeg = algos.DHFA(f=f, eta=1 / L**0.5, lam=lam, integrator="extg", aggfunc="avg")
N, K = algos.make_NK_for_dhfa(
    eta=1 / L**0.5, total_iters=args.total_iters, alpha=m, lam=lam
)
x_dfhaeg_sol, f_vals_dhfaeg = dhfaeg.run(x_0=x_0, K=K, N=N)

# we know the solution here
min_val = f(
    torch.linalg.solve(X.T @ X + args.N * args.beta * torch.eye(args.d), X.T @ y)
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(
    range(0, args.total_iters + 1),
    (f_vals_gd - min_val).clamp_min_(torch.tensor(0.0)),
    label="GD",
    lw=3.5,
    alpha=0.9,
)
ax.plot(
    range(0, args.total_iters + 1),
    (f_vals_agd - min_val).clamp_min_(torch.tensor(0.0)),
    label="AGD",
    lw=3.5,
    alpha=0.9,
)
ax.plot(
    [0] + np.cumsum(N).tolist(),
    (f_vals_dhfaeg - min_val).clamp_min_(torch.tensor(0.0)),
    label="dHFA-eg",
    marker="o",
    lw=3.5,
    markersize=10,
    alpha=0.9,
    markeredgecolor="grey",
    markevery=2,
)
ax.set_yscale("log")
ax.set_xlabel(r"\# of iterations / integration steps")
ax.set_ylabel(r"$f(x_{K}) - f^{\star}$")
ax.set_ylim(bottom=1e-15)
plt.tight_layout()
plt.savefig(
    f"linear-regression-beta={args.beta},N={args.N},d={args.d},seed={args.seed}.pdf"
)
