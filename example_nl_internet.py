"""
Fit a parametric generalized Nash equilibrium problem (GNEP) with a shared
inequality constraint from cost samples, as described in [1, Section 6.3].

We compare two choices for the nonlinear potential term added on top of a convex
quadratic model: an input-convex NN potential (potential={'type': 'NN', ...}) 
and no potential term (potential=None, i.e., a pure quadratic approximation, [1, Lemma 4.2]).

The underlying game is the "internet switching" model of Facchinei, Kanzow
(2009), Example A.1:

    J_i(x,q) = -x_i/(x_1+...+x_N)*(1-(x_1+...+x_N)*q),  i=1,...,N_PLAYERS
        x_1+...+x_N <= 1/q,  x_i >= LB_X

with q as the game's parameter (the original problem uses p=1/q as the
parameter). After fitting, each learned model is compared to the true game on
two held-out sets: best-response samples (br_errors, agent-wise) and GNEs
computed at N_EQUILIBRIA held-out parameter values via nashopt.GNEP
(EquilibriumSolver), warm-started from the true equilibrium (eq_errors).

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
from tqdm import tqdm

from monotone_games import (
    CostSampleLoss,
    EquilibriumSolver,
    GameLearner,
    NESampleRegularizer,
    StructuredMonotoneCost,
)

# ########################################
SEED = 3 # baseline random seed for reproducibility

N_PLAYERS = 5 # number of players
DIMS = [1] * N_PLAYERS # dimension of each player's decision variable
NX = sum(DIMS) # total dimension of all players' decision variables
NPAR = 1 # number of parameters q of the game
LB_X = 0.01 # per-player lower bound on x_i (x_i >= LB_X)
QMIN = 0.5 # lower bound of the q sampling range
QMAX_DIVISOR = LB_X * N_PLAYERS # qmax = 1/QMAX_DIVISOR (feasibility limit N_PLAYERS*LB_X*qmax <= 1)

MU = 0. # (strong) monotonicity constant used to fit the game
RHO = 1.e-8 # L2 regularization of the cost-model parameters
# POTENTIALS = [ #(label, Aq, potential, phi) choices to compare
#     ("Affine",            "affine", None, None),
#     ("Neural",            {'type': 'NN', 'layers': [5,5], 'activation': jax.nn.swish}, None, None),
#     ("Potential",         {'type': 'NN', 'layers': [5,5], 'activation': jax.nn.swish},
#      {'type': 'NN', 'layers': [5, 5],'activation': jax.nn.softplus},
#      None # no phi_i term
#      #'quadratic'  # quadratic phi_i term
#     ),
# ] # (label, potential, phi) StructuredMonotoneCost configurations to fit and compare
POTENTIALS = [
    ("none", "affine", None, 'quadratic'),
    ("NN", "affine", {'type': 'NN', 'layers': [20, 20], 'activation': jax.nn.softplus}, #None # no phi_i term
     'quadratic'  # quadratic phi_i term
    ),
] # (label, potential, phi) StructuredMonotoneCost configurations to fit and compare

USE_NE_DATA = False # use NE-sample data as an extra regularizer (off: no constraints in the training data)

N_TRAIN = 1000 * (N_PLAYERS - 1) # number of training cost samples
N_VAL = 1000 * (N_PLAYERS - 1) # number of validation samples, used to choose the best model among parallel training runs
N_TEST = 1000 * (N_PLAYERS - 1) # number of held-out cost-test samples, used to report R2
N_EQUILIBRIA = 50 # number of held-out parameter values at which the learned model's GNE is computed
N_BR_TEST = 200 # number of held-out best-response test samples

ADAM_EPOCHS = 1000 # initial ADAM iterations during training
LBFGS_EPOCHS = 5000 # L-BFGS iterations used for training
N_SEEDS = cpu_count() # number of parallel training runs
# ########################################


def build_problem():
    """Build the fixed problem data (parameter box, cost/best-response
    functions) for the internet switching model."""
    lb = LB_X * np.ones(N_PLAYERS)
    qmin = QMIN * np.ones(NPAR)
    qmax = (1. / QMAX_DIVISOR) * np.ones(NPAR)

    @jax.jit
    def Ji(xi, x_mi, q):
        """J_i(x,q) for player i, given its own xi and the others' x_mi."""
        s = xi + jnp.sum(x_mi)
        return -xi / s * (1 - s * q)

    def best_response(i, x, q):
        """argmin_{x_i} J_i(x,q), given x_1+...+x_N <= 1/q, x_i >= LB_X.
        Returns (xi, feasible, J_i at that xi)."""
        x_mi = jnp.concatenate([x[:i], x[i + 1:]])
        a = jnp.sum(x_mi)
        if lb[i] + a > 1. / q:
            return x[i], False, Ji(x[i], x_mi, q)
        xi = jnp.maximum(jnp.sqrt(a / q) - a, lb[i])
        return xi, True, Ji(xi, x_mi, q)

    return SimpleNamespace(lb=lb, qmin=qmin, qmax=qmax, Ji=Ji, best_response=best_response)


def _sample_feasible(rng, n_samples, prob):
    """Sample q uniformly and a feasible x on x_1+...+x_N <= 1/q, x_i >= LB_X
    (via a Dirichlet split of the slack 1/q - sum(lb))."""
    P = np.linspace(1./prob.qmax, 1./prob.qmin, num=n_samples, endpoint=True).reshape(n_samples, 1)
    Q = 1. / P
    if 1:
        # Dirichlet split of the slack, matching the sampling used in [1]
        slack = 1. / Q[:, 0] - prob.lb.sum()
        w = rng.dirichlet(np.ones(N_PLAYERS + 1), size=n_samples)
        X = slack[:, None] * w[:, :N_PLAYERS] + prob.lb[None, :]
    else:
        # simpler: sample x_i uniformly in [LB_X, 1/q] and project onto the shared constraint set
        X = rng.uniform(prob.lb, 1. / Q, size=(n_samples, N_PLAYERS))
        for k in range(n_samples):
            if np.sum(X[k]) > 1. / Q[k, 0]:
                X[k] = prob.lb + (X[k] - prob.lb) * (1. / Q[k, 0] - prob.lb.sum()) / (np.sum(X[k]) - prob.lb.sum())                
    return X, Q


def true_costs(x, q, prob):
    """Return [J_1(x,q),...,J_N(x,q)] for one sample."""
    return np.array([float(prob.Ji(x[i], jnp.concatenate((x[:i], x[i + 1:])), q[0]))
                      for i in range(N_PLAYERS)])


def sample_cost_data(rng, n_samples, prob):
    """Sample a feasible x,q and evaluate the true cost J_i(x,q) for every
    player at the sampled x."""
    X, Q = _sample_feasible(rng, n_samples, prob)
    J = np.array([true_costs(x, q, prob) for x, q in zip(X, Q)])
    return X, Q, J


def true_equilibrium(q, x0, prob):
    """Solve the true game's GNE at parameter q via nashopt.GNEP applied to
    the true cost functions J_i(x,q), warm-started from the feasible x0.

    No closed form: the symmetric unconstrained solution x_i=(N-1)/(N^2 q)
    can violate x_i>=LB_X once q is large enough that the shared budget 1/q
    gets tight, so the lower bound (and/or the shared constraint) may be
    active at the true GNE.
    """
    from nashopt import GNEP
    f = [lambda x, i=i: prob.Ji(x[i], jnp.concatenate((x[:i], x[i + 1:])), q[0])
         for i in range(N_PLAYERS)]
    gnep = GNEP(sizes=DIMS, f=f, g=lambda x: shared_constraint(x, q), ng=1, lb=prob.lb)
    sol = gnep.solve(jnp.asarray(x0), verbose=0, solver='hybr')
    return sol.x


def sample_equilibria(rng, n_samples, prob):
    """Sample q and the corresponding true GNE x*(q) (true_equilibrium),
    warm-started from a feasible point."""
    X0, Q = _sample_feasible(rng, n_samples, prob)
    X = np.array([true_equilibrium(q, x0, prob) for q, x0 in zip(Q, X0)])
    return X, Q


def sample_br_data(rng, n_samples, prob):
    """Sample q, a feasible x_{-i}, and a random agent i per sample, then
    replace x_i with agent i's true best response to x_{-i} (closed form,
    prob.best_response) -- best-response test data (Xbr,Q,agent_idx)
    analogous to example_quad_game.py's sample_br_data."""
    X, Q = _sample_feasible(rng, n_samples, prob)
    agent_idx = rng.integers(0, N_PLAYERS, size=n_samples)
    for k in range(n_samples):
        xi, _, _ = prob.best_response(agent_idx[k], X[k], Q[k, 0])
        X[k, agent_idx[k]] = float(xi)
    return X, Q, agent_idx


def sample_best_response_data(rng, n_samples, prob):
    """Sample a feasible x,q as in sample_cost_data, then replace each cost
    column i with the value J_i attains at player i's own best response to
    x_{-i} -- used as NE-sample regularizer data (USE_NE_DATA)."""
    X, Q = _sample_feasible(rng, n_samples, prob)

    pairs = [(i, k) for i in range(N_PLAYERS) for k in range(n_samples)]

    def _cost_at_br(i, k):
        _, _, Ji_val = prob.best_response(i, X[k], Q[k, 0])
        return float(Ji_val)

    values = Parallel(n_jobs=-1)(
        delayed(_cost_at_br)(i, k)
        for i, k in tqdm(pairs, desc=f"N={N_PLAYERS} best-response data",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"))
    J = np.empty((n_samples, N_PLAYERS))
    for (i, k), v in zip(pairs, values):
        J[k, i] = v
    return X, Q, J


def r2_scores(y, y_hat):
    """Return one R^2 value per player and their average."""
    residual = np.sum((y - y_hat) ** 2, axis=0)
    total = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    scores = 1.0 - residual / total
    return scores, float(np.mean(scores))


def predict_costs(cost_model, theta, X, Q):
    def one_cost(x, q):
        return jnp.stack(cost_model.costs(x, q, theta))

    return np.asarray(jax.jit(jax.vmap(one_cost))(jnp.asarray(X), jnp.asarray(Q)))


def fit_model(train_data, val_data, data_ne, ne_regularizer, potential, phi, Aq):
    """Fit a StructuredMonotoneCost model with the given potential and phi_i
    to train_data."""
    cost_model = StructuredMonotoneCost(DIMS, NPAR, mu=MU, potential=potential, phi=phi, Aq=Aq)
    learner = GameLearner(cost_model, CostSampleLoss(), ne_regularizer=ne_regularizer, rho=RHO)
    result = learner.fit(
        train_data, data_ne=data_ne,
        adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
        seeds=np.arange(SEED, SEED + N_SEEDS), val_data=val_data)
    return cost_model, result


@jax.jit
def shared_constraint(x, q):
    """x_1+...+x_N <= 1/q."""
    return jnp.sum(x) - 1. / q


def equilibrium_diagnostics(cost_model, theta, prob, X_eq, Q_eq):
    """Solve the learned model's GNE at each held-out q, warm-started from
    the true equilibrium X_eq (also an excellent initial point for the
    learned model), and report the KKT residual and the error against the
    true equilibrium.

    EquilibriumSolver.solve (nashopt.GNEP under the hood, non-variational:
    each agent i gets its own multipliers) finds x*(q) together with
    lambda_i* >= 0 per agent for the shared constraint g(x,q) =
    x_1+...+x_N - 1/q <= 0 and mu_i* >= 0 per agent for its bound
    x_i >= LB_X, solving the stacked KKT system r(x,lambda,mu) = 0 with, for
    i=1,...,N_PLAYERS:

        stationarity:    dJ_i/dx_i(x,q) + lambda_i - mu_i = 0
        complementarity: phi_FB(lambda_i, -g(x,q)) = 0
                         phi_FB(mu_i, x_i - LB_X) = 0

    where phi_FB(a,b) = sqrt(a^2+b^2) - a - b is the Fischer-Burmeister NCP
    function (phi_FB(a,b) = 0 iff a >= 0, b >= 0, a*b = 0), enforcing
    lambda_i, mu_i >= 0 together with lambda_i*(-g(x,q)) = 0 and
    mu_i*(x_i-LB_X) = 0.

    EquilibriumSolver.solve() JIT-compiles the KKT residual/Jacobian ONCE
    per (instance, theta) -- with q a genuine JAX-traced argument, not a
    value baked into a fresh Python closure -- and reuses that compiled
    executable for every subsequent call with the SAME theta object, so
    calling it n_samples times below (q is the only thing that changes
    across the held-out points) costs one compile plus n_samples cheap
    root-finds, not n_samples full retrace-and-recompiles. shared_constraint
    already has the (x,q) signature this expects for g(x,p), so it is
    passed directly (no per-sample closure needed). The explicit
    compile_parametric(theta) call below is optional -- solve() would
    compile automatically on its first call regardless -- it is only here
    to report compile time separately.

    "KKT residual" is residual_norms[j] = ||sol.res||_2, the root-finder's
    residual r(x*,lambda*,mu*) (0 at an exact equilibrium; a measure of
    solve accuracy, not a distance in x-space). "eq_errors" is
    ||x*_hat(Q_eq[j]) - X_eq[j]||_2, the Euclidean distance between the
    learned model's GNE and the true game's GNE X_eq[j] at the same
    parameter Q_eq[j].
    """
    n_samples = X_eq.shape[0]
    residual_norms = np.zeros(n_samples)
    eq_errors = np.zeros(n_samples)
    eq_solver = EquilibriumSolver(cost_model, constraints={
        'g': shared_constraint, 'ng': 1, 'lb': prob.lb})
    t0 = time.time()
    eq_solver.compile_parametric(theta)
    compile_time = time.time() - t0
    t0 = time.time()
    for j in range(n_samples):
        q = Q_eq[j]
        sol = eq_solver.solve(q, theta, x0=X_eq[j], verbose=0, solver='hybr')
        residual_norms[j] = np.linalg.norm(sol.res)
        eq_errors[j] = np.linalg.norm(sol.x - X_eq[j])
        print(f"Sample {j + 1}/{n_samples}: q = {q[0]:5.2f}, "
              f"eq error = {eq_errors[j]:7.4e}, "
              f"||KKT residual|| = {residual_norms[j]:5.2g}")
    elapsed = time.time() - t0
    return SimpleNamespace(residual_norms=residual_norms, eq_errors=eq_errors,
                           elapsed=elapsed, compile_time=compile_time)


def best_response_diagnostics(cost_model, theta, prob, Xbr, Qbr, agent_idx, verbose=False):
    """Compute the learned model's best response (EquilibriumSolver.
    best_response, honoring the shared/box constraints) at each test sample
    (Xbr[k],Qbr[k],agent_idx[k]) and compare it to the true best response
    stored in Xbr[k,agent_idx[k]] (sample_br_data); br_errors[k] =
    ||x_i_hat - x_i_true||_2. Analogous to example_quad_game.py's br_errors,
    but via the general constrained best-response solve since the learned
    model here need not be quadratic.

    EquilibriumSolver.best_response JIT-compiles, once per (instance,
    theta), the value-and-gradient of the penalized best-response objective
    for every agent -- with agent i/x/q all genuine JAX-traced (or static,
    for i) arguments, not baked into a fresh Python closure -- and reuses
    those compiled executables for every subsequent call with the SAME
    theta object, so calling it n_samples times below (most of which differ
    in agent index and/or q) costs one compile plus n_samples cheap
    L-BFGS-B solves. shared_constraint already has the (x,q) signature this
    expects for g(x,p), so it is passed directly (no per-sample closure
    needed). The explicit compile_parametric_br(theta) call below is
    optional -- best_response() would compile automatically on its first
    call regardless -- it is only here to compile before the threaded
    Parallel() below fires (the compiled functions it reads are pre-built
    and side-effect-free, so sharing eq_solver across threads is safe)."""
    n_samples = Xbr.shape[0]
    par = Parallel(n_jobs=-1, backend="threading", verbose=10 if verbose else 0)

    eq_solver = EquilibriumSolver(cost_model, constraints={
        'g': shared_constraint, 'ng': 1, 'lb': prob.lb})
    eq_solver.compile_parametric_br(theta)

    def _learned_br(k):
        i, q = int(agent_idx[k]), Qbr[k]
        sol = eq_solver.best_response(q, theta, i, Xbr[k])
        return sol.x[cost_model.isi[i]]

    t0 = time.time()
    xi_hat = np.array(par(delayed(_learned_br)(k) for k in range(n_samples)))
    elapsed = time.time() - t0
    xi_true = np.array([Xbr[k, cost_model.isi[int(agent_idx[k])]] for k in range(n_samples)])
    br_errors = np.linalg.norm(xi_hat - xi_true, axis=1)
    return SimpleNamespace(br_errors=br_errors, elapsed=elapsed)


def model_diagnostics(name, potential_label, result, eq_result, br_result,
                      X_test, Q_test, J_test):
    """Report training, best-response, and equilibrium-solve diagnostics for
    the fitted model."""
    cost_model, theta = result.cost_model, result.theta
    J_hat = predict_costs(cost_model, theta, X_test, Q_test)
    _, mean_r2 = r2_scores(J_test, J_hat)
    print(f"\n{name}")
    print(f"  training time                 : {result.training_time:8.4f} s")
    print(f"  held-out cost R2 (mean over players): {mean_r2:8.2%}")
    print(f"  parametric GNEP compile time   : {eq_result.compile_time:8.4f} s")
    print(f"  equilibrium solve time        : {eq_result.elapsed:8.4f} s")
    print(f"  KKT residual (mean / max)     : {eq_result.residual_norms.mean():8.4e} / "
          f"{eq_result.residual_norms.max():8.4e}")
    print(f"  held-out BR error (mean / max) : {br_result.br_errors.mean():8.4e} / "
          f"{br_result.br_errors.max():8.4e}")
    print(f"  learned-NE error (mean / max)  : {eq_result.eq_errors.mean():8.4e} / "
          f"{eq_result.eq_errors.max():8.4e}")

    return {
        "name": name,
        "potential": potential_label,
        "cpu_time": result.training_time,
        "fit_r2": mean_r2,
        "eq_time": eq_result.elapsed,
        "residual_max": float(eq_result.residual_norms.max()),
        "br_error_mean": float(br_result.br_errors.mean()),
        "eq_error_mean": float(eq_result.eq_errors.mean()),
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
    """Return a LaTeX table summarizing the learned GNE models' diagnostics,
    one row per potential choice (NN potential vs. no potential)."""
    header = r"  potential & time (s) & R$^2$ (\%) & BR error & NE error \\"
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \renewcommand{\arraystretch}{1.} % Adjust row separation",
        r"  \caption{Learned generalized Nash equilibrium models for the internet switching game "
        f"({N_PLAYERS} players).}}",
        r"  \label{tab:nl_internet_game}",
        r"  \begin{tabular}{c|rrrr}",
        r"  \hline",
        header,
        r"  \hline",
    ]
    best = {
        "cpu_time": min(r["cpu_time"] for r in rows),
        "fit_r2": max(r["fit_r2"] for r in rows),
        "br_error_mean": min(r["br_error_mean"] for r in rows),
        "eq_error_mean": min(r["eq_error_mean"] for r in rows),
    }
    for r in rows:
        cpu_time_s = f"{r['cpu_time']:.4f}"
        r2_s = f"{100.0 * r['fit_r2']:.2f}"
        cells = [
            r["potential"],
            r"\textbf{" + cpu_time_s + "}" if r["cpu_time"] == best["cpu_time"] else cpu_time_s,
            r"\textbf{" + r2_s + "}" if r["fit_r2"] == best["fit_r2"] else r2_s,
            _latex_sci(r["br_error_mean"], bold=(r["br_error_mean"] == best["br_error_mean"])),
            _latex_sci(r["eq_error_mean"], bold=(r["eq_error_mean"] == best["eq_error_mean"])),
        ]
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(SEED)
prob = build_problem()

train_data = sample_cost_data(rng, N_TRAIN, prob)
val_data = sample_cost_data(rng, N_VAL, prob)
X_test, Q_test, J_test = sample_cost_data(rng, N_TEST, prob)

if USE_NE_DATA:
    X_ne, Q_ne, J_ne = sample_best_response_data(rng, N_EQUILIBRIA, prob)
    data_ne = (X_ne, Q_ne, J_ne)
    ne_regularizer = NESampleRegularizer(lam1=1.0, lam2=1.0, match='cost')
else:
    data_ne, ne_regularizer = None, None

print("\nTrue game")
print(f"  q sampling range               : [{prob.qmin[0]:.4f}, {prob.qmax[0]:.4f}]")
print(f"  per-player lower bound         : {LB_X:.4f}")

print(f"\nComputing {N_EQUILIBRIA} held-out true equilibria and "
      f"{N_BR_TEST} best-response test samples...")
X_eq, Q_eq = sample_equilibria(rng, N_EQUILIBRIA, prob)
Xbr_test, Qbr_test, ibr_test = sample_br_data(rng, N_BR_TEST, prob)

rows = []
for label, Aq, potential, phi in POTENTIALS:
    print(f"\nFitting model {len(rows) + 1}/{len(POTENTIALS)}: potential={label!r}")
    t0 = time.time()
    cost_model, result = fit_model(train_data, val_data, data_ne, ne_regularizer, potential, phi, Aq=Aq)
    print(f"Total fitting time: {time.time() - t0:.2f} s")

    eq_result = equilibrium_diagnostics(cost_model, result.theta, prob, X_eq, Q_eq)
    br_result = best_response_diagnostics(cost_model, result.theta, prob, Xbr_test, Qbr_test, ibr_test)
    rows.append(model_diagnostics(
        f"cost-based fit ({label})", label, result, eq_result, br_result,
        X_test, Q_test, J_test))

print("\n\n% ===== LaTeX Table (learned GNE model comparison) =====\n")
print(make_latex_table(rows))
