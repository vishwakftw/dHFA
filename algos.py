import math
import torch
from typing import List, Callable, Tuple


class Algorithm:
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], eta: float):
        self._f = f
        self._eta = eta

    def run(self, x_0: torch.Tensor, K: int, **kwargs):
        raise NotImplementedError

    def grad_f(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x.detach().requires_grad_(True)
        val = self._f(x_)
        val.backward()
        return x_.grad.detach()

    def set_eta(self, eta: float):
        self._eta = eta


class DHFA(Algorithm):
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        eta: float,
        lam: float,
        integrator: str,
        aggfunc: str,
    ):
        super().__init__(f=f, eta=eta)
        self.lam = lam

        # Integrator settings
        match integrator:
            case "extg":
                self._integ = self._extg_step
            case "exp":
                self._integ = self._exp_step
            case "leap":
                self._integ = self._leap_step
            case _:
                raise ValueError("Invalid integrator")

        # Aggregator settings
        self._aggfunc = aggfunc
        match aggfunc:
            case "avg":
                self._weights = lambda n, N: N - n + 1
            case "s-avg":
                self._weights = lambda n, N: 1
            case _:
                raise ValueError("Invalid aggregator")

    def run(self, x_0: torch.Tensor, K: int, N: List[int]):
        if self._eta is None:
            raise ValueError("step size unset")
        assert len(N) == K

        x_prev = x_0.clone().detach()
        f_vals = [self._f(x_0).item()]

        for i in range(K):
            N_i = N[i]

            # initial conditions
            x_n = x_prev.clone()
            y_n = torch.zeros_like(x_prev)

            x_avg = torch.zeros_like(x_prev)

            # integrate and aggregate
            for n in range(1, N_i + 1):
                x_n, y_n = self._integ(x_n, y_n)
                x_avg = x_avg + self._weights(n, N_i) * x_n

            # normalise
            fact = sum(self._weights(n, N_i) for n in range(1, N_i + 1))
            x_avg /= fact

            # lambda weights
            x_prev = (1.0 / (self.lam + 1.0)) * (x_avg + self.lam * x_n)
            f_vals.append(self._f(x_prev).item())

        return x_prev, torch.tensor(f_vals)

    def _extg_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x' = (x + η y) - η² ∇f(x + η y)
        x_plus_ey = x + self._eta * y
        g1 = self.grad_f(x_plus_ey)
        x_new = x_plus_ey - self._eta**2 * g1

        # y' = y - η ∇f(x')
        g2 = self.grad_f(x_new)
        y_new = y - self._eta * g2

        return x_new, y_new

    def _exp_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x' = (x + η y)
        # y' = y - η ∇f(x)
        x_new = x + self._eta * y
        y_new = y - self._eta * self.grad_f(x)
        return x_new, y_new

    def _leap_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g1 = self.grad_f(x)
        x_new = x + self._eta * y - 0.5 * g1 * self._eta**2
        g2 = self.grad_f(x_new)
        y_new = y - 0.5 * (g1 + g2) * self._eta
        return x_new, y_new


class GD(Algorithm):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], eta: float):
        super().__init__(f=f, eta=eta)

    def run(self, x_0: torch.Tensor, K: int):
        if self._eta is None:
            raise ValueError("step size unset")

        x = x_0.clone().detach()
        f_vals = [self._f(x_0).item()]

        for k in range(K):
            g = self.grad_f(x)
            x = x - self._eta * g
            f_vals.append(self._f(x).item())

        return x, torch.tensor(f_vals)


class AGD(Algorithm):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], eta: float):
        super().__init__(f=f, eta=eta)

    def run(self, x_0: torch.Tensor, K: int, momentum: float | None = None):
        if self._eta is None:
            raise ValueError("step size unset")

        f_vals = [self._f(x_0).item()]

        x_prev = x_0.clone().detach()  # x_{k-1}
        x_curr = x_0.clone().detach()  # x_k

        for k in range(1, K + 1):
            # momentum coefficient
            # if momentum is unfixed, then we do the default
            gamma = (k - 1.0) / (k + 2.0) if momentum is None else momentum

            # lookahead
            y = x_curr + gamma * (x_curr - x_prev)

            # gradient step
            g = self.grad_f(y)
            x_next = y - self._eta * g

            # shift
            x_prev = x_curr
            x_curr = x_next
            f_vals.append(self._f(x_curr).item())

        return x_curr, torch.tensor(f_vals)


def make_NK_for_dhfa(
    eta: float, total_iters: int, alpha: float, lam: float | None = None
):
    # problem is strongly convex
    if alpha > 0:
        Nsing = int((4 / alpha) ** 0.5 / eta)
        N = []
        while (val := sum(N)) < total_iters:
            newN = Nsing
            N.append(newN - max(0, val + newN - total_iters))
        K = len(N)
    else:
        N = [4]
        # contraction parameter from the paper
        q = ((6 * lam * (1 + lam)) / (6 * lam**2 + 4 * lam + 1)) ** 0.5
        while (val := sum(N[1:])) < total_iters:
            newN = int(math.ceil(q * N[-1] + 1 / 2))
            N.append(newN - max(0, val + newN - total_iters))
        N = N[1:]
        K = len(N)
    return N, K
