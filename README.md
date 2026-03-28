#### dHFA

Code to reproduce the experiments in the paper: **Accelerated Convex Optimization via Hamiltonian Dynamics with Deterministic Integration Time.**

Packages used:
- PyTorch
- NumPy
- Matplotlib (for plotting)

##### Code files

- `algos.py` contains the implementation of the proposed algorithm along with gradient descent and accelerated gradient descent baselines.
- `utils.py` contains code to generate the data and perform grid search over the step size.
- `main_linear_regression.py` and `main_logistic_regression.py` can be used to reproduce the plots in Figure 5.1 with default seed.
- `main_avg_variation.py` can be used to reproduce the plots in Figure 5.2 with default seed.
- `main_intg_variation.py` can be used to reproduce the plots in Figure 5.3 with default seed.

These scripts require options to be passed to them; these can be found by running `python <script> --help`.

##### Citation

TODO
