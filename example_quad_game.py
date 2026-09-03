"""
Fit a parametric quadratic Nash equilibrium problem on data generated
from random monotone quadratic games, as described in [1, Section 6.1]. 

We use two alternative data sources:

1. cost samples J_i(x,p), fit via GameLearner/StructuredMonotoneCost
   (the generic bilevel/autodiff route);
2. best-response samples of a single agent given the others' decisions, fit
   via QuadraticInverseSolver's closed-form SDP/LS+SDP fast paths (Sec. 5)
   plus its NLS fast path, fit via jax_sysid from cpu_count() parallel
   initial seeds (one of them warm-started from the LS+SDP solution, the
   rest random).

Both approaches recover a StructuredMonotoneCost(potential=None) quadratic
game model theta, compared on a common held-out best-response test set and
a common Nash-equilibrium test set.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 

(C) 2026 A. Bemporad
"""

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from joblib import cpu_count, delayed, Parallel
import numpy as np

from monotone_games import (
    CostSampleLoss,
    EquilibriumSolver,
    GameLearner,
    NESampleRegularizer,
    QuadraticInverseSolver,
    StructuredMonotoneCost,
)

# ########################################
# Select any subset of the learning approaches.
FIT_BASED_ON_COSTS = True
FIT_INVERSE_LEARNING = True

SEED = 3 # baseline random seed for reproducibility

N_PLAYERS = 4 # number of players
DIM = 2 # dimension of each player's decision variable
DIMS = [DIM] * N_PLAYERS # dimension of each player's decision variable
NX = sum(DIMS) # total dimension of all players' decision variables
NPAR = 2 # number of parameters p of the game
LOWER_X = -2. # lower bound of the x sampling box
UPPER_X = 2. # upper bound of the x sampling box
LOWER_P = -1. # lower bound of the p sampling box
UPPER_P = 1. # upper bound of the p sampling box

MU = 0. # (strong) monotonicity constant, both to generate and to fit the game

N_TRAIN = 500 # number of training samples (cost samples, or best-response samples)
N_VAL = 100 # number of validation samples, used to choose the best model among parallel training runs
N_TEST = 200 # number of held-out test samples (cost samples, or best-response samples)
N_EQUILIBRIA = 50 # number of NE samples used to evaluate learned models

RHO = 1.e-8 # L2 regularization of the cost-model parameters
RHO_NLS = 1.e-12 # L2 regularization of the cost-model parameters for NLS (inverse-learning only)
POTENTIAL = None # StructuredMonotoneCost potential used by FIT_BASED_ON_COSTS
#POTENTIAL = {'type': 'NN', 'layers': [3, 2], 'activation': jax.nn.softplus}
USE_NE_DATA = False # use NE-sample data as an extra regularizer for FIT_BASED_ON_COSTS
ADAM_EPOCHS = 1000 # initial ADAM iterations during training
LBFGS_EPOCHS = 5000 # L-BFGS iterations used for training
N_SEEDS = cpu_count() # number of parallel training runs (FIT_BASED_ON_COSTS,
                      # FIT_INVERSE_LEARNING's NLS fit)
# ########################################


def build_true_game(rng):
    """Build a random quadratic game shifted to have (strong) monotonicity
    constant MU: Q[i] (nx x nx, only rows si:ei used by player i), c1[i]
    (nx x npar), c2[i] (nx,). G collects the player-own row blocks of Q, so
    the unconstrained NE solves F(x,p) = G@x + q(p) = 0."""
    Q = [rng.standard_normal((NX, NX)) for _ in range(N_PLAYERS)]
    Q = [B.T @ B for B in Q]

    def own_blocks(Q):
        G = np.zeros((NX, NX))
        for i in range(N_PLAYERS):
            si, ei = i * DIM, (i + 1) * DIM
            G[si:ei, :] = Q[i][si:ei, :]
        return G

    G = own_blocks(Q)
    shift = MU - np.linalg.eigvalsh(0.5 * (G + G.T)).min()
    for i in range(N_PLAYERS):
        si, ei = i * DIM, (i + 1) * DIM
        Q[i][si:ei, si:ei] += shift * np.eye(DIM)
    G = own_blocks(Q)
    lam_min = np.linalg.eigvalsh(0.5 * (G + G.T)).min()
    print(f"Monotonicity check: lambda_min(0.5*(G+G.T)) = {lam_min:.4e} "
          f"(shift applied: {shift:.4e})")

    c1 = [rng.standard_normal((NX, NPAR)) for _ in range(N_PLAYERS)]
    c2 = [rng.standard_normal(NX) for _ in range(N_PLAYERS)]
    return Q, G, c1, c2


def true_costs(x, p, Q, c1, c2):
    """Return [J_1(x,p),...,J_N(x,p)] for one sample."""
    return np.array([(0.5 * Q[i] @ x + c1[i] @ p + c2[i]) @ x
                      for i in range(N_PLAYERS)])


def true_q(p, c1, c2):
    """Stationarity offset q(p): F(x,p) = G@x + q(p) = 0 at the NE."""
    q = np.zeros(NX)
    for i in range(N_PLAYERS):
        si, ei = i * DIM, (i + 1) * DIM
        q[si:ei] = (c1[i] @ p + c2[i])[si:ei]
    return q


def true_best_response(x, p, i, Q, c1, c2):
    """argmin_{x_i} J_i(x,p), holding x_{-i} fixed."""
    si, ei = i * DIM, (i + 1) * DIM
    others = np.r_[0:si, ei:NX]
    rhs = Q[i][si:ei, others] @ x[others] + (c1[i] @ p + c2[i])[si:ei]
    return -np.linalg.solve(Q[i][si:ei, si:ei], rhs)


def sample_cost_data(rng, n_samples, Q, c1, c2):
    """Sample x,p independently and uniformly from their boxes."""
    X = rng.uniform(LOWER_X, UPPER_X, size=(n_samples, NX))
    P = rng.uniform(LOWER_P, UPPER_P, size=(n_samples, NPAR))
    J = np.array([true_costs(x, p, Q, c1, c2) for x, p in zip(X, P)])
    return X, P, J


def sample_br_data(rng, n_samples, Q, c1, c2):
    """Sample best-response data: for each k, pick a random agent i_k, a
    random x_{k,-i} and p_k, and complete x_k with agent i_k's best response."""
    X = rng.uniform(LOWER_X, UPPER_X, size=(n_samples, NX))
    P = rng.uniform(LOWER_P, UPPER_P, size=(n_samples, NPAR))
    idx = rng.integers(0, N_PLAYERS, size=n_samples)
    for k in range(n_samples):
        X[k, idx[k] * DIM:(idx[k] + 1) * DIM] = true_best_response(
            X[k], P[k], idx[k], Q, c1, c2)
    return X, P, idx


def sample_equilibria(rng, n_samples, G, c1, c2):
    """Sample p and the corresponding exact unconstrained NE x*(p)."""
    P = rng.uniform(LOWER_P, UPPER_P, size=(n_samples, NPAR))
    X = np.array([-np.linalg.solve(G, true_q(p, c1, c2)) for p in P])
    return X, P


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


def learned_best_response(cost_model, theta, x, p, i):
    """Closed-form best response of a StructuredMonotoneCost(potential=None)
    model: argmin_{x_i} J_i(x,p;theta) = -A_ii^{-1}(A_{i,-i} x_{-i} + q_i)."""
    theta_quad = theta[:cost_model.n_param_quad]
    A, q = cost_model.pseudogradient_matrix(jnp.asarray(p), theta_quad)
    A, q = np.asarray(A), np.asarray(q)
    isi, nisi = cost_model.isi[i], cost_model.nisi[i]
    return -np.linalg.solve(A[np.ix_(isi, isi)], A[np.ix_(isi, nisi)] @ x[nisi] + q[isi])


def model_diagnostics(name, result, X_test, P_test, J_test,
                      Xbr_test, Pbr_test, ibr_test, X_eq, P_eq, verbose=False):
    """Report cost R2 (informational only), held-out best-response error, and
    NE error -- the latter two are directly comparable across both learning
    approaches, unlike R2 (best-response data carries no information about
    the x_{-i}-only part of the true cost, so inverse-learning models are not
    expected to reconstruct J_i itself)."""
    cost_model, theta = result.cost_model, result.theta
    J_hat = predict_costs(cost_model, theta, X_test, P_test)
    _, mean_r2 = r2_scores(J_test, J_hat)

    eq_solver = EquilibriumSolver(cost_model)
    # threading backend: shares the parent process's jax_enable_x64 config
    # (loky worker processes do not inherit it, see cost_models.py's
    # monotonicity_check for the same gotcha).
    par = Parallel(n_jobs=-1, backend="threading", verbose=10 if verbose else 0)
    br_errors = np.array(par(
        delayed(learned_best_response)(cost_model, theta, x, p, i)
        for x, p, i in zip(Xbr_test, Pbr_test, ibr_test)))
    br_errors = np.linalg.norm(
        br_errors - np.array([x[cost_model.isi[i]] for x, i in zip(Xbr_test, ibr_test)]),
        axis=1)
    eq_errors = np.array(par(
        delayed(eq_solver.solve_quadratic)(p, theta) for p in P_eq))
    eq_errors = np.linalg.norm(
        np.array([r.x for r in eq_errors]) - X_eq, axis=1)

    print(f"\n{name}")
    print(f"  training time                 : {result.training_time:8.4f} s")
    print(f"  held-out cost R2 (informational): {mean_r2:8.2%}")
    print(f"  held-out BR error (max)       : {br_errors.max():8.4e}")
    print(f"  learned-NE error (max)        : {eq_errors.max():8.4e}")

    return {
        "cpu_time": result.training_time,
        "br_error_mean": float(br_errors.mean()),
        "br_error_max": float(br_errors.max()),
        "eq_error_mean": float(eq_errors.mean()),
        "eq_error_max": float(eq_errors.max()),
    }


def _latex_sci(v, bold=False):
    """Format v as LaTeX scientific notation, e.g. $1.23\\times10^{-4}$."""
    if v is None or np.isnan(v):
        return "--"
    mantissa, exponent = f"{v:.2e}".split("e")
    body = f"{mantissa} \\times 10^{{{int(exponent)}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


def make_latex_table(rows):
    """Return a LaTeX table summarizing the learned-model diagnostics, one
    row per learning approach (comparable across FIT_BASED_ON_COSTS and
    FIT_INVERSE_LEARNING: both are evaluated on the same held-out
    best-response and NE test sets)."""
    method_names = {
        "cost-based fit": r"NLS",
        "inverse-SDP": "SDP",
        "inverse-LS+SDP": "LS+SDP",
        "inverse-NLS": "NLS",
    }
    data_names = {
        "cost-based fit": r"$J_{i,k}$",
        "inverse-SDP": r"$\bar x_{i,k}$",
        "inverse-LS+SDP": r"$\bar x_{i,k}$",
        "inverse-NLS": r"$\bar x_{i,k}$",
    }
    header = r"  method & data & time (s) & BR error & NE error \\"
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \renewcommand{\arraystretch}{1.} % Adjust row separation",
        r"  \caption{Comparison of learning approaches on the parametric quadratic game.}",
        r"  \label{tab:quad_game}",
        r"  \begin{tabular}{l|c|rrr}",
        r"  \hline",
        header,
        r"  \hline",
    ]
    best = {
        "cpu_time": min(r["cpu_time"] for r in rows),
        "br_error_mean": min(r["br_error_mean"] for r in rows),
        "eq_error_mean": min(r["eq_error_mean"] for r in rows),
    }
    for r in rows:
        cells = [method_names[r["name"]], data_names[r["name"]]]
        s = f"{r['cpu_time']:.4f}"
        cells.append(r"\textbf{" + s + "}"
                     if r["cpu_time"] == best["cpu_time"] else s)
        cells.append(_latex_sci(r["br_error_mean"], bold=(r["br_error_mean"] == best["br_error_mean"])))
        cells.append(_latex_sci(r["eq_error_mean"], bold=(r["eq_error_mean"] == best["eq_error_mean"])))
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def wrap_result(cost_model, theta, training_time):
    return SimpleNamespace(cost_model=cost_model, theta=theta, training_time=training_time)


def fit_models(rng, train_data, val_data, val_data_br, true_game):
    """Fit the learning approaches enabled by the global flags."""
    Q, G, c1, c2 = true_game
    n_models = (1 if FIT_BASED_ON_COSTS else 0) + (3 if FIT_INVERSE_LEARNING else 0)
    if n_models == 0:
        raise ValueError("At least one FIT_* learning-approach flag must be True")
    results = []

    if FIT_BASED_ON_COSTS:
        print(f"\nFitting model {len(results) + 1}/{n_models}: "
              "cost-based fit from cost samples")
        cost_model = StructuredMonotoneCost(DIMS, NPAR, mu=MU, Aq='constant', potential=POTENTIAL)
        if USE_NE_DATA:
            X_e, P_e = sample_equilibria(rng, N_TRAIN, G, c1, c2)
            J_e = np.array([true_costs(x, p, Q, c1, c2) for x, p in zip(X_e, P_e)])
            data_ne = (X_e, P_e, J_e)
            ne_regularizer = NESampleRegularizer(lam1=1.0, lam2=1.0, match='cost')
        else:
            data_ne, ne_regularizer = None, None
        learner = GameLearner(cost_model, CostSampleLoss(),
                              ne_regularizer=ne_regularizer, rho=RHO)
        result = learner.fit(
            train_data, data_ne=data_ne,
            adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
            seeds=np.arange(SEED, SEED + N_SEEDS), val_data=val_data)
        results.append(("cost-based fit", result))
        # Print total number of model coefficients learned (informational only)
        print(f"  learned model has {sum(x.size for x in jax.tree_util.tree_leaves(result.model.params))} coefficients")

    if FIT_INVERSE_LEARNING:
        Xbr, Pbr, ibr = sample_br_data(rng, N_TRAIN, Q, c1, c2)
        inv = QuadraticInverseSolver(DIMS, NPAR, mu=MU)

        print(f"\nFitting model {len(results) + 1}/{n_models}: "
              "SDP from best-response data")
        t0 = time.time()
        res_sdp = inv.fit_sdp(Xbr, Pbr, ibr, solver='SCS')
        results.append(("inverse-SDP", wrap_result(*inv.to_cost_model(res_sdp), time.time() - t0)))

        print(f"Fitting model {len(results) + 1}/{n_models}: "
              "LS+SDP from best-response data")
        t0 = time.time()
        res_cascade = inv.fit_ls_sdp_cascade(Xbr, Pbr, ibr, solver='SCS')
        results.append(("inverse-LS+SDP", wrap_result(*inv.to_cost_model(res_cascade), time.time() - t0)))

        # NLS fit via jax_sysid from N_SEEDS parallel seeds: one warm-started
        # from the cascade solution (rho=1e-12, see fit_nls's docstring for
        # why), the rest random; the seed with lowest held-out best-response
        # residual (val_data_br) is kept.
        print(f"Fitting model {len(results) + 1}/{n_models}: "
              "NLS from cpu_count() parallel seeds, best-response data")
        t0 = time.time()
        res_nls = inv.fit_nls(
            Xbr, Pbr, ibr, seeds=np.arange(SEED, SEED + N_SEEDS),
            warm_start=res_cascade, val_data=val_data_br, rho=RHO_NLS,
            adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS)
        results.append(("inverse-NLS", wrap_result(*inv.to_cost_model(res_nls), time.time() - t0)))

    return results


if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(SEED)
true_game = build_true_game(rng)
Q, G, c1, c2 = true_game

train_data = sample_cost_data(rng, N_TRAIN, Q, c1, c2)
val_data = sample_cost_data(rng, N_VAL, Q, c1, c2)
val_data_br = sample_br_data(rng, N_VAL, Q, c1, c2)
test_data = sample_cost_data(rng, N_TEST, Q, c1, c2)
Xbr_test, Pbr_test, ibr_test = sample_br_data(rng, N_TEST, Q, c1, c2)
X_eq, P_eq = sample_equilibria(rng, N_EQUILIBRIA, G, c1, c2)

X_test, P_test, J_test = test_data
print("\nTrue game")
print(f"  x sampling box                : [{LOWER_X:.4f}, {UPPER_X:.4f}]^{NX}")
print(f"  p sampling box                : [{LOWER_P:.4f}, {UPPER_P:.4f}]^{NPAR}")
print(f"  requested (strong) monotonicity mu: {MU:8.4e}")

t0 = time.time()
results = fit_models(rng, train_data, val_data, val_data_br, true_game)
print(f"\nTotal fitting time: {time.time() - t0:.2f} s")

rows = []
for i, (name, result) in enumerate(results, start=1):
    metrics = model_diagnostics(
        f"{i}. {name}", result, X_test, P_test, J_test,
        Xbr_test, Pbr_test, ibr_test, X_eq, P_eq)
    metrics["name"] = name
    rows.append(metrics)

print("\n\n% ===== LaTeX Table (learning-approach comparison) =====\n")
print(make_latex_table(rows))
