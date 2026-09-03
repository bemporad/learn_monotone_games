"""
Fit the parametric two-player monotone-game counterexample in Section 6.2 of [1].

The true costs are

    J1(x,p) = 1/2 (x1-p1)^2 + (x1-p1) sin(x2-p2),
    J2(x,p) = 1/2 (x2-p2)^2 - (x2-p2) sin(x1-p1).

Four learned models are compared:

1. monotonicity by construction, using an input-convex NN potential Psi,
   an NN parameterization of the affine pseudogradient terms (including the
   parameter-dependent offset t(p), called q(p) by StructuredMonotoneCost),
   and an NN phi_i(x_-i,p) for each player;
2. input-convex-in-x_i costs J_i, with monotonicity promoted by penalizing
   pairwise violations of the monotonicity inequality of the pseudogradient
   itself, (x_j-x_h)^T (F(x_j,p_j)-F(x_h,p_j)) >= mu ||x_j-x_h||^2;
3. input-convex-in-x_i costs J_i, with monotonicity promoted by penalizing
   negative eigenvalues of the symmetric pseudogradient Jacobian;
4. input-convex-in-x_i costs J_i, with the symmetric pseudogradient Jacobian
   matched to the Hessian of an auxiliary input-convex neural network.

x is sampled uniformly from [-pi/2, pi/2]^2 and p from [-pi/4, pi/4]^2.  The
exact unconstrained Nash equilibrium is x*(p) = p.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 


(C) 2026 A. Bemporad
"""

import time

import jax
import jax.numpy as jnp
from joblib import cpu_count
import numpy as np
from scipy.optimize import root

from monotone_games import (
    AuxiliaryPotentialPenalty,
    CostSampleLoss,
    GameLearner,
    JacobianEigPenalty,
    PenalizedConvexCost,
    StructuredMonotoneCost,
    ViolationPenalty,
)

# ########################################
# Select any subset of the four learning approaches.
FIT_MONOTONE_BY_CONSTRUCTION = True
FIT_PSEUDOGRADIENT_VIOLATION_PENALTY = True
FIT_JACOBIAN_EIG_PENALTY = True
FIT_AUXILIARY_ICNN_PENALTY = True

SEED = 3 # baseline random seed for reproducibility

DIMS = [1, 1] # dimension of each player's decision variable
NX = sum(DIMS) # total dimension of all players' decision variables
NPAR = 2 # number of parameters p of the game
LOWER_X = -np.pi/2. # lower bound of the x sampling box
UPPER_X = np.pi/2. # upper bound of the x sampling box
LOWER_P = -np.pi/4. # lower bound of the p sampling box
UPPER_P = np.pi/4. # upper bound of the p sampling box

MU = 0.2 # requested (strong) monotonicity constant

N_TRAIN = 2000 # number of training samples
N_VAL = 1000 # number of validation samples, used to choose the best model among parallel training runs
N_TEST = 2000 # number of held-out test samples
N_EQUILIBRIA = 50 # number of NE samples used to evaluate learned models
N_MONOTONICITY = 2000 # unlabelled samples used by the monotonicity penalties
N_VIOLATION_SAMPLES = 50 # monotonicity samples used by the pairwise pseudogradient-violation penalty (O(M^2) pairs)

WIDTH = [4, 4] # Ji model NN layers
WIDTH_AUX_ICNN = [4, 4] # layers of the auxiliary ICNN used when FIT_AUXILIARY_ICNN_PENALTY = True
GAMMA = 1.e3 # monotonicity penalty weight
RHO = 1.e-8 # L2 regularization of the cost-model parameters
ADAM_EPOCHS = 1000 # initial ADAM iterations during training
LBFGS_EPOCHS = 5000 # L-BFGS iterations used for training
N_SEEDS = cpu_count() # number parallel training runs
GLOBOPT_MAXEVAL = 2000 # nlopt evaluations per parallel rectangle in monotonicity_check
# ########################################

def true_costs(x, p):
    """Return [J1(x,p), J2(x,p)] for one sample (JAX traceable)."""
    y = x - p
    return jnp.array([
        0.5 * y[0] ** 2 + y[0] * jnp.sin(y[1]),
        0.5 * y[1] ** 2 - y[1] * jnp.sin(y[0]),
    ])


def true_pseudogradient(x, p):
    """Return F=col(dJ1/dx1,dJ2/dx2) for one sample."""
    y = x - p
    return jnp.array([y[0] + jnp.sin(y[1]), y[1] - jnp.sin(y[0])])


def sample_cost_data(rng, n_samples):
    """Sample x,p independently and uniformly from their boxes."""
    X = rng.uniform(LOWER_X, UPPER_X, size=(n_samples, NX))
    P = rng.uniform(LOWER_P, UPPER_P, size=(n_samples, NPAR))
    J = np.asarray(jax.vmap(true_costs)(jnp.asarray(X), jnp.asarray(P)))
    return X, P, J


def r2_scores(y, y_hat):
    """Return one R^2 value per player and their average."""
    residual = np.sum((y - y_hat) ** 2, axis=0)
    total = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    scores = 1.0 - residual / total
    return scores, float(np.mean(scores))


def predict_costs(cost_model, theta, X, P):
    def one_cost(x, p):
        return jnp.stack(cost_model.costs(x, p, theta))

    return np.asarray(jax.jit(jax.vmap(one_cost))(jnp.asarray(X), jnp.asarray(P)))


def model_diagnostics(name, result, X_test, P_test, J_test, P_eq):
    """Report cost, pseudogradient, monotonicity (sampled and box-certified),
    and equilibrium errors."""
    cost_model, theta = result.cost_model, result.theta
    J_hat = predict_costs(cost_model, theta, X_test, P_test)
    r2, mean_r2 = r2_scores(J_test, J_hat)

    def learned_F(x, p):
        return cost_model.pseudogradient(x, p, theta)

    learned_F = jax.jit(learned_F)
    F_hat = np.asarray(jax.vmap(learned_F)(jnp.asarray(X_test), jnp.asarray(P_test)))
    F_true = np.asarray(jax.vmap(true_pseudogradient)(
        jnp.asarray(X_test), jnp.asarray(P_test)))
    f_rmse = float(np.sqrt(np.mean((F_hat - F_true) ** 2)))

    def min_sym_eig(x, p):
        jac = cost_model.pseudogradient_jacobian(x, p, theta)
        return jnp.linalg.eigvalsh(0.5 * (jac + jac.T))[0]

    eig = np.asarray(jax.jit(jax.vmap(min_sym_eig))(
        jnp.asarray(X_test), jnp.asarray(P_test)))
    violation_fraction = float(np.mean(eig < MU - 1.e-7))

    eq_errors = []
    root_failures = 0
    for p in P_eq:
        # The true equilibrium p is also an excellent initial point for the
        # learned equilibrium.  scipy.root avoids an external nashopt path.
        sol = root(lambda x: np.asarray(learned_F(jnp.asarray(x), jnp.asarray(p)),
                                        dtype=float), p, method="hybr")
        if not sol.success:
            root_failures += 1
        eq_errors.append(np.linalg.norm(sol.x - p))
    eq_errors = np.asarray(eq_errors)

    # Box-certified monotonicity test: global minimization of the smallest
    # eigenvalue of the symmetric pseudogradient Jacobian over the sampling box.
    t_check = time.perf_counter()
    mono = cost_model.monotonicity_check(
        theta,
        LOWER_X * np.ones(NX), UPPER_X * np.ones(NX),
        LOWER_P * np.ones(NPAR), UPPER_P * np.ones(NPAR),
        mu=MU, maxeval=GLOBOPT_MAXEVAL)
    t_check = time.perf_counter() - t_check

    print(f"\n{name}")
    print(f"  training time                         : {result.training_time:8.2f} s")
    print(f"  held-out R2 (J1, J2 / average)        : "
          f"{r2[0]:8.2%}, {r2[1]:8.2%} / {mean_r2:8.2%}")
    print(f"  held-out pseudogradient RMSE          : {f_rmse:8.4e}")
    print(f"  sampled min symmetric-Jacobian eig    : {eig.min():8.4e}")
    print(f"  sampled mu-monotonicity violation rate: {violation_fraction:8.2%}")
    lam_min_str = f"{mono.lam_min:8.4e}"
    if mono.lam_min < 0:
        lam_min_str = f"\033[1;31m{lam_min_str}\033[0m"
    print(f"  certified min symmetric-Jacobian eig  : {lam_min_str} "
          f"({t_check:.2f} s)")
    print(f"    attained at x = {np.array2string(mono.x, precision=4)}, "
          f"p = {np.array2string(mono.p, precision=4)}")
    print(f"  mu-monotone over the whole box        : {mono.is_monotone}")
    print(f"  learned-NE error mean / max           : "
          f"{eq_errors.mean():8.4e} / {eq_errors.max():8.4e}")
    if root_failures:
        print(f"  warning: root solver did not converge : {root_failures}/{len(P_eq)}")

    return {
        "cpu_time": result.training_time,
        "sampled_lam_min": float(eig.min()),
        "global_lam_min": float(mono.lam_min),
        "fit": mean_r2,
        "eq_error_mean": float(eq_errors.mean()),
    }


def _latex_num(v, bold=False):
    """Format v in LaTeX fixed-point notation, e.g. $12.3456$."""
    if v is None or np.isnan(v):
        return "--"
    body = f"{v:.4f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


def make_latex_table(rows):
    """Return a LaTeX table summarizing the learned-model diagnostics,
    one row per learning approach."""
    header = (r"  method & time (s) & "
              r"$\lambda_{\rm min}$ (test/global) & R$^2$ (\%) & NE error \\")
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.} % Adjust row separation",
        r"  \caption{Comparison of learning approaches on example~\eqref{eq:parametric-counterexample}.}",
        r"      \label{tab:counterexample}",
        r"  \begin{tabular}{l|rrrr}",
        r"  \hline",
        header,
        r"  \hline",
    ]
    best = {
        "cpu_time": min(r["cpu_time"] for r in rows),
        "sampled_lam_min": max(r["sampled_lam_min"] for r in rows),
        "global_lam_min": max(r["global_lam_min"] for r in rows),
        "fit": max(r["fit"] for r in rows),
        "eq_error_mean": min(r["eq_error_mean"] for r in rows),
    }
    for r in rows:
        cells = [r["name"]]
        s = f"{r['cpu_time']:.2f}"
        cells.append(r"\textbf{" + s + "}"
                     if r["cpu_time"] == best["cpu_time"] else s)
        cells.append(" / ".join(
            _latex_num(r[key], bold=(r[key] == best[key]))
            for key in ("sampled_lam_min", "global_lam_min")))
        s = f"{100.0 * r['fit']:.2f}"
        cells.append(r"\textbf{" + s + "}" if r["fit"] == best["fit"] else s)
        cells.append(_latex_num(r["eq_error_mean"],
                                bold=(r["eq_error_mean"] == best["eq_error_mean"])))
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def fit_models(train_data, val_data, monotonicity_data):
    """Fit the learning approaches enabled by the global flags."""
    activation = jax.nn.softplus
    seeds = np.arange(SEED, SEED + N_SEEDS)
    n_enabled = sum((FIT_MONOTONE_BY_CONSTRUCTION,
                     FIT_PSEUDOGRADIENT_VIOLATION_PENALTY,
                     FIT_JACOBIAN_EIG_PENALTY,
                     FIT_AUXILIARY_ICNN_PENALTY))
    if n_enabled == 0:
        raise ValueError("At least one FIT_* learning-approach flag must be True")
    results = []

    if FIT_MONOTONE_BY_CONSTRUCTION:
        # In the package, q(p) is the affine pseudogradient offset, denoted
        # t(p) here; Aq='NN' jointly emits C(p), D(p), t(p).
        structured_model = StructuredMonotoneCost(
            DIMS,
            NPAR,
            mu=MU,
            potential={
                "type": "NN",
                "layers": [WIDTH[0], WIDTH[1]],
                "activation": activation,
            },
            Aq={
                "type": "NN",
                "layers": [WIDTH[0], WIDTH[1]],
                "activation": activation,
            },
            phi={
                "type": "NN",
                "layers": [WIDTH[0], WIDTH[1]],
                "activation": activation,
            },
        )
        learner = GameLearner(
            structured_model, CostSampleLoss(), rho=RHO)
        print(f"\nFitting model {len(results) + 1}/{n_enabled}: "
              "monotone by construction (ICNN Psi, NN t and phi_i)")
        result = learner.fit(
            train_data,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=LBFGS_EPOCHS,
            seeds=seeds,
            val_data=val_data,
        )
        results.append((r"monotonic~\eqref{eq:J_i-form}", result))

    if FIT_PSEUDOGRADIENT_VIOLATION_PENALTY:
        violation_model = PenalizedConvexCost(
            DIMS,
            NPAR,
            own_layers=(WIDTH[0], WIDTH[1]),
            coupling_layers=(WIDTH[0], WIDTH[1]),
            activation=activation,
        )
        penalty = ViolationPenalty(gamma=GAMMA, mu=MU)
        learner = GameLearner(
            violation_model,
            CostSampleLoss(),
            monotonicity_penalty=penalty,
            rho=RHO,
        )
        # L_M1 compares all ordered sample pairs, so its cost grows with the
        # square of the monotonicity-set size: use a smaller subset here.
        X_m, P_m = monotonicity_data
        n_pairs = min(N_VIOLATION_SAMPLES, X_m.shape[0])
        print(f"\nFitting model {len(results) + 1}/{n_enabled}: "
              "player-wise ICNN costs with pseudogradient-violation penalty "
              f"({n_pairs} samples)")
        result = learner.fit(
            train_data,
            data_m=(X_m[:n_pairs], P_m[:n_pairs]),
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=LBFGS_EPOCHS,
            seeds=seeds,
            val_data=val_data,
        )
        results.append((r"ICNN + $\LL_{M1}$~\eqref{eq:monotonicity-violation-1}", result))

    if FIT_JACOBIAN_EIG_PENALTY:
        penalized_model = PenalizedConvexCost(
            DIMS,
            NPAR,
            own_layers=(WIDTH[0], WIDTH[1]),
            coupling_layers=(WIDTH[0], WIDTH[1]),
            activation=activation,
        )
        penalty = JacobianEigPenalty(gamma=GAMMA, mu=MU)
        learner = GameLearner(
            penalized_model,
            CostSampleLoss(),
            monotonicity_penalty=penalty,
            rho=RHO,
        )
        print(f"\nFitting model {len(results) + 1}/{n_enabled}: "
              "player-wise ICNN costs with symmetric-Jacobian penalty")
        result = learner.fit(
            train_data,
            data_m=monotonicity_data,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=LBFGS_EPOCHS,
            seeds=seeds,
            val_data=val_data,
        )
        results.append((r"ICNN + $\LL_{M2}$~\eqref{eq:monotonicity-violation-2}", result))

    if FIT_AUXILIARY_ICNN_PENALTY:
        auxiliary_model = PenalizedConvexCost(
            DIMS,
            NPAR,
            own_layers=(WIDTH[0], WIDTH[1]),
            coupling_layers=(WIDTH[0], WIDTH[1]),
            activation=activation,
        )
        penalty = AuxiliaryPotentialPenalty(
            gamma=GAMMA,
            phi_layers=(WIDTH_AUX_ICNN[0], WIDTH_AUX_ICNN[1]),
            activation=activation,
            mu=MU,
        )
        learner = GameLearner(
            auxiliary_model,
            CostSampleLoss(),
            monotonicity_penalty=penalty,
            rho=RHO,
        )
        print(f"\nFitting model {len(results) + 1}/{n_enabled}: "
              "player-wise ICNN costs with auxiliary-ICNN Jacobian penalty")
        result = learner.fit(
            train_data,
            data_m=monotonicity_data,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_epochs=LBFGS_EPOCHS,
            seeds=seeds,
            val_data=val_data,
        )
        results.append((r"ICNN + $\LL_{M3}$~\eqref{eq:monotonicity-violation-3}", result))

    return results


if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(SEED)
train_data = sample_cost_data(rng, N_TRAIN)
val_data = sample_cost_data(rng, N_VAL)
test_data = sample_cost_data(rng, N_TEST)
X_m = rng.uniform(LOWER_X, UPPER_X, size=(N_MONOTONICITY, NX))
P_m = rng.uniform(LOWER_P, UPPER_P, size=(N_MONOTONICITY, NPAR))
P_eq = rng.uniform(LOWER_P, UPPER_P, size=(N_EQUILIBRIA, NPAR))

# Analytically, lambda_min(sym nabla F) =
# 1 - |cos(x2-p2)-cos(x1-p1)|/2 >= 0.
X_test, P_test, J_test = test_data
y_test = X_test - P_test
true_eig = 1.0 - 0.5 * np.abs(np.cos(y_test[:, 1]) - np.cos(y_test[:, 0]))
print("True game")
print(f"  x sampling box                        : "
        f"[{LOWER_X:.4f}, {UPPER_X:.4f}]^2")
print(f"  p sampling box                        : "
        f"[{LOWER_P:.4f}, {UPPER_P:.4f}]^2")
print(f"  sampled min symmetric-Jacobian eig    : {true_eig.min():8.4e}")
print("  exact unconstrained equilibrium       : x*(p) = p")
print(f"  requested surrogate monotonicity mu   : {MU:8.4e}")

t0 = time.perf_counter()
results = fit_models(train_data, val_data, (X_m, P_m))
print(f"\nTotal fitting time: {time.perf_counter() - t0:.2f} s")

rows = []
for i, (name, result) in enumerate(results, start=1):
    metrics = model_diagnostics(f"{i}. {name}", result,
                                X_test, P_test, J_test, P_eq)
    metrics["name"] = name
    rows.append(metrics)

print("\n\n% ===== LaTeX Table (learning-approach comparison) =====\n")
print(make_latex_table(rows))
