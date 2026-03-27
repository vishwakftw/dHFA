import numpy as np
import torch

from argparse import ArgumentParser
from matplotlib import pyplot as plt

import algos
import utils

torch.set_default_dtype(torch.float64)

plt.rcParams.update({"text.usetex": True, "font.size": 20, "axes.grid": True})

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
p.add_argument(
    "--expt_type", type=str, choices=["lin-reg", "log-reg"], help="Experiment type"
)
args = p.parse_args()

x_0 = torch.zeros(args.d)
match args.expt_type:
    case "lin-reg":
        f, X, y = utils.make_linear_regression(
            N=args.N, d=args.d, alpha=args.alpha, seed=args.seed
        )
        eigvals_XTX = torch.linalg.eigvalsh(X.T @ X).clamp_min_(torch.tensor(0.0))
        L, m = (
            eigvals_XTX.max() / args.N + args.alpha,
            eigvals_XTX.min() / args.N + args.alpha,
        )

    case "log-reg":
        f, X, y = utils.make_logistic_regression(
            N=args.N, d=args.d, alpha=args.alpha, seed=args.seed
        )
        L, m = 1, args.alpha
    case _:
        raise ValueError

# setting parameter for dHFA
if m > 0:
    lam = 0
else:
    lam = args.lam

# make schedule
N, K = algos.make_NK_for_dhfa(
    eta=1 / L**0.5, total_iters=args.total_iters, alpha=m, lam=lam
)

dhfaeg = algos.DHFA(f=f, eta=1 / L**0.5, lam=lam, integrator="extg", aggfunc="avg")
x_dfhaeg_sol, f_vals_dhfaeg = dhfaeg.run(x_0=x_0, K=K, N=N)

dhfaex = algos.DHFA(f=f, eta=1 / L**0.5, lam=lam, integrator="exp", aggfunc="avg")
x_dfhaex_sol, f_vals_dhfaex = dhfaex.run(x_0=x_0, K=K, N=N)

dhfalf = algos.DHFA(f=f, eta=1 / L**0.5, lam=lam, integrator="leap", aggfunc="avg")
x_dfhalf_sol, f_vals_dhfalf = dhfalf.run(x_0=x_0, K=K, N=N)

if args.expt_type == "lin-reg":
    min_val = f(
        torch.linalg.solve(X.T @ X + args.N * args.alpha * torch.eye(args.d), X.T @ y)
    )
    ylims = {"bottom": 1e-15, "top": 1e03}

else:
    gd = algos.GD(f=f, eta=1 / L)
    x_gd_sol, f_vals_gd = gd.run(x_0=x_0, K=args.total_iters)

    agd = algos.AGD(f=f, eta=1 / L)
    momentum = None
    if m > 0:
        momentum = (1 - (m / L) ** 0.5) / (1 + (m / L) ** 0.5)

    x_agd_sol, f_vals_agd = agd.run(x_0=x_0, K=args.total_iters, momentum=momentum)

    min_val = min(
        min(f_vals_dhfaeg),
        min(f_vals_dhfaex),
        min(f_vals_dhfalf),
        min(f_vals_gd),
        min(f_vals_agd),
    )
    ylims = {"bottom": 1e-5}

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(
    [0] + np.cumsum(N).tolist(),
    (f_vals_dhfaeg - min_val).clamp_min_(torch.tensor(0.0)),
    label="dHFA-eg",
    marker=".",
    lw=3.5,
    markersize=12,
    alpha=0.7,
)
ax.plot(
    [0] + np.cumsum(N).tolist(),
    (f_vals_dhfaex - min_val).clamp_min_(torch.tensor(0.0)),
    label="dHFA-ex",
    marker=".",
    lw=3.5,
    markersize=12,
    alpha=0.7,
)
ax.plot(
    [0] + np.cumsum(N).tolist(),
    (f_vals_dhfalf - min_val).clamp_min_(torch.tensor(0.0)),
    label="dHFA-lf",
    marker=".",
    lw=3.5,
    markersize=12,
    alpha=0.7,
)

ax.legend()
ax.set_yscale("log")
ax.set_xlabel("Number of iterations $K$")
ax.set_ylabel(r"$f(x_{K}) - f^{\star}$")
ax.set_ylim(**ylims)
plt.tight_layout()
plt.savefig(
    f"intg-var-{args.expt_type}-alpha={args.alpha},N={args.N},d={args.d},seed={args.seed}.pdf"
)
