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
p.add_argument("--alpha", type=float, help="Regularisation parameter")
p.add_argument("--seed", type=int, default=1729, help="Seed for reproducibility")
p.add_argument("--total_iters", type=int, help="Number of iterations")
p.add_argument("--lam", type=float, default=(1 + 3**0.5) / 2, help="Lambda parameter")
args = p.parse_args()

x_0 = torch.zeros(args.d)
f, X, y = utils.make_logistic_regression(
    N=args.N, d=args.d, alpha=args.alpha, seed=args.seed
)

etas = torch.logspace(-5, 0, 25).numpy().tolist()

gd = algos.GD(f=f, eta=None)
best_eta_gd, _ = utils.find_best_eta(
    etas=etas,
    alg=gd,
    total_iters=args.total_iters,
    alpha=args.alpha,
    x_0=x_0,
)
gd.set_eta(eta=best_eta_gd)
x_gd_sol, f_vals_gd = gd.run(x_0=x_0, K=args.total_iters)

agd = algos.AGD(f=f, eta=None)
best_eta_agd, _ = utils.find_best_eta(
    etas=etas,
    alg=agd,
    total_iters=args.total_iters,
    alpha=args.alpha,
    x_0=x_0,
)
agd.set_eta(best_eta_agd)
momentum = None
if args.alpha > 0:
    momentum = (1 - (args.alpha * best_eta_agd) ** 0.5) / (
        1 + (args.alpha * best_eta_agd) ** 0.5
    )

x_agd_sol, f_vals_agd = agd.run(x_0=x_0, K=args.total_iters, momentum=momentum)

if args.alpha > 0:
    lam = 0
else:
    lam = args.lam
dhfaeg = algos.DHFA(f=f, eta=None, lam=lam, integrator="extg", aggfunc="avg")
best_eta_dhfaeg, _ = utils.find_best_eta(
    etas=etas,
    alg=dhfaeg,
    total_iters=args.total_iters,
    alpha=args.alpha,
    x_0=x_0,
    lam=lam,
)
dhfaeg.set_eta(best_eta_dhfaeg)
N, K = algos.make_NK_for_dhfa(
    eta=best_eta_dhfaeg, total_iters=args.total_iters, alpha=args.alpha, lam=lam
)
x_dfhaeg_sol, f_vals_dhfaeg = dhfaeg.run(x_0=x_0, K=K, N=N)

min_val = min(min(f_vals_gd), min(f_vals_agd), min(f_vals_dhfaeg))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(
    range(0, args.total_iters + 1),
    f_vals_gd - min_val,
    label="GD",
    lw=3.5,
    alpha=0.9,
)
ax.plot(
    range(0, args.total_iters + 1),
    f_vals_agd - min_val,
    label="AGD",
    lw=3.5,
    alpha=0.9,
)
ax.plot(
    [0] + np.cumsum(N).tolist(),
    f_vals_dhfaeg - min_val,
    label="dHFA-eg",
    marker=".",
    lw=3.5,
    markersize=10,
    alpha=0.9,
)
ax.set_yscale("log")
ax.set_xlabel("Number of iterations $K$")
ax.set_ylabel(r"$f(x_{K}) - f^{\star}$")
ax.set_ylim(1e-5)
plt.tight_layout()
plt.savefig(
    f"logistic-regression-alpha={args.alpha},N={args.N},d={args.d},seed={args.seed}.pdf"
)
