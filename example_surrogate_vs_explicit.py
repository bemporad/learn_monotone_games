"""
Compare two ways of getting a fast parametric solution p -> x*(p) to a random
monotone LQ-GNEP against the ground-truth GNE:

1. mpfit.LQ_GNEP: a neural-network explicit solution p -> x*(p), fit directly
   on best-response data.
2. StructuredMonotoneCost (Aq='constant', potential=None): a monotone-by-
   construction quadratic-game surrogate, fit on cost samples J_i(x,p) via
   GameLearner/CostSampleLoss (see example_quad_game.py), then solved online
   at each test p with the 'daqp' KKT-QP method (qp_gnep).

Both use the same underlying game and the same (X,P) sample locations for
training (the latter obtained from mpfit's own data generator, evaluated with
the true cost functions to get the J_i(x,p) samples needed by CostSampleLoss).
Both are evaluated on the same test parameters p (mpfit's own res.P_test)
against:

    - best-response error   E[||xhat - BR(xhat)||]  (mean of UNSQUARED
                             per-sample norms, feasible BRs only, matching
                             the convention/definition used by the other
                             examples in example_quad_game.py/
                             example_nl_internet.py/example_cstr_control.py)
    - constraint violation  mean/max of A x - b - S p (and Aeq x - beq)
    - GNE error             ||xhat(p) - x_true(p)||, x_true(p) solving the
                             TRUE LQ-GNEP (Q,c,F,A,b,S,E,h) via daqp/qp_gnep
    - cost/value fit R2     on a held-out validation set, using each
                             method's own Stage-1 target: mpfit's value net
                             Jhat_i(x_{-i},p) fits the best-response VALUE
                             min_{x_i} f_i(x,p); the surrogate's cost model
                             fits the raw cost J_i(x,p) directly

The first two metrics reuse mpfit's own evaluate() (same self.A/b_lin/S/lb/ub
and self.f), so all three solutions are scored under identical definitions.

Solving the TRUE LQ-GNEP at every test p (via qp_gnep, Part 3) is itself an
extra QP solve per point; the COMPARE_GNE_GROUND_TRUTH flag makes this
optional. When False, X_true/stats_true/the GNE-error metrics and the Part 4b
pseudogradient diagnostic are all skipped, and only best-response error,
constraint violation, and cost/value fit R2 are reported.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 
    
(C) 2026 A. Bemporad
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
from joblib import cpu_count
from jax_sysid.utils import compute_scores

from nashopt.lq.generate_random import generate_random  # requires nashopt>=1.3.0
from nashopt.lq.qp_gnep import qp_gnep  # requires nashopt>=1.3.0

from mpfit import LQ_GNEP
from monotone_games import CostSampleLoss, GameLearner, StructuredMonotoneCost

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

np.random.seed(0)

# =====================================================================
# Part 1: random monotone parametric LQ-GNEP, and mpfit's explicit
# solution p -> x*(p) (same problem/setup as example_lq_gne.py /
# nashopt.lq.generate_random)
# =====================================================================

N = 3      # number of agents
n = 2      # number of decision variables per agent
m = 16     # number of shared inequality constraints  A x <= b
q = 0      # number of shared equality constraints     E x = h
m_act = m//4  # number of active inequalities at the reference point x_star
mu = 0.1   # monotone_eps for the random game generation

npar = 2       # number of parameters
N_train = 2000 # number of training samples
N_val = 1000   # number of validation samples
N_test = 1000  # number of test samples

COMPARE_GNE_GROUND_TRUTH = False  # solve/compare vs the TRUE LQ-GNEP via
                                  # qp_gnep (Part 3/4/4b) -- an extra QP
                                  # solve per test point; False skips it and
                                  # only reports best-response error,
                                  # constraint violation, and cost/value fit R2

nvar = N * n
sizes = [n] * N

gnep_true, data = generate_random(dim=sizes, m=m, m_act=m_act, q=q, seed=0, mu=mu)

Q_true = gnep_true.Q  # list of N matrices, shape (nvar, nvar), only row-block i used
c_true0 = gnep_true.c  # list of N vectors, shape (nvar,), only entries [i*n:(i+1)*n] used
A_true = gnep_true.A  # shape (m, nvar)
b_true0 = gnep_true.b  # shape (m,)
E_true = gnep_true.Aeq  # shape (q, nvar)
h_true = gnep_true.beq  # shape (q,)

G_true = data["G"]  # true pseudogradient, generally asymmetric
mu_check = np.linalg.eigvalsh(0.5 * (G_true + G_true.T)).min()
print(f"Monotonicity check (generate_random): "
      f"lambda_min(0.5*(G+G.T)) = {mu_check:.4e}")

# Parametric dependence on top of the fixed game: F couples the linear cost
# terms, S couples the shared-inequality RHS; equalities stay fixed (no p).
F = [.5*np.round(np.random.randn(nvar, npar), 2) for _ in range(N)]
S_ineq = .1 + np.round(0.1 * np.random.rand(m, npar), 2)  # strictly positive

# Scale the cost functions by a constant to affect fit quality
# const = 1.
# Q_true = [Q / const for Q in Q_true]
# c_true0 = [c / const for c in c_true0]
# F = [F_i / const for F_i in F]

pmin =-1.*np.ones(npar)
pmax = 1.*np.ones(npar)

lb = -np.inf * np.ones(nvar)  # no variable bounds
ub = np.inf * np.ones(nvar)

# The game itself stays unconstrained (lb/ub above), but x_star (and hence
# the game's natural operating range) is O(1) (generate_random draws it
# standard normal); without a box, mpfit's internal data generator falls
# back to sampling x in [-100,100]^nvar, which inflates best-response value
# targets far beyond the game's natural scale and makes Stage-1 value-
# function fitting needlessly hard. xlb_sample/xub_sample below narrow only
# the *training-data sampling* box, decoupled from lb/ub.
xlb_sample = -10.0 * np.ones(nvar)
xub_sample = 10.0 * np.ones(nvar)

# mpfit.LQ_GNEP only supports shared inequalities A x <= b + S p, so fold
# E x = h into two inequalities with zero parametric coupling.
A_comb = np.vstack((A_true, E_true, -E_true))
b_comb = np.concatenate((b_true0, h_true, -h_true))
S_comb = np.vstack((S_ineq, np.zeros((q, npar)), np.zeros((q, npar))))
ncon = m + 2 * q

value_nn = [30, 20]
nn_value_act = jax.nn.swish
n1, n2 = 30, 20
solution_nn_act = jax.nn.relu

mp = LQ_GNEP(N, sizes, npar, Q_true, c_true0, F, A_comb, b_comb, S_comb, lb, ub, pmin, pmax, ncon,
             n1=n1, n2=n2, value_nn=value_nn, nn_value_act=nn_value_act, solution_nn_act=solution_nn_act, convexity_nn=False, beta=1000,
             xlb_sample=xlb_sample, xub_sample=xub_sample)
mp.setup(N_train=N_train, N_val=N_val)
mp.solve()
res = mp.test(N_test=N_test)

P_test = res.P_test
X_mpfit = res.X_hat

# =====================================================================
# Part 2: fit a StructuredMonotoneCost surrogate on cost samples, using the
# same (X,P) sample locations mpfit generated internally for training/
# validation, but with cost-sample targets J_i(x,p) (needed by
# CostSampleLoss) instead of mpfit's best-response *value* targets.
# =====================================================================

print("\nRegenerating mpfit's (X,P) training/validation samples "
      "(same seeds as mp.setup()) as cost-sample data for the surrogate fit...")
X_train, P_train, _ = mp._generate_br_data(N_train, seed=4)  # mp.setup()'s default seed_train
X_val, P_val, J_val_br = mp._generate_br_data(N_val, seed=5)  # mp.setup()'s default seed_val;
# J_val_br = best-response VALUES min_{x_i} f_i(x,p), the target mpfit's own
# Stage-1 value nets (self.th1) were fit against -- kept below for the R2
# comparison instead of being discarded.

J_train = np.array([[float(mp.f[i](x, p)) for i in range(N)] for x, p in zip(X_train, P_train)])
J_val = np.array([[float(mp.f[i](x, p)) for i in range(N)] for x, p in zip(X_val, P_val)])

MU = 0.           # We do not assume we know the mu of the true game
RHO = 1.e-8       # L2 regularization on the cost-model parameters
ADAM_EPOCHS = 1000
LBFGS_EPOCHS = 5000
SEED = 3
N_SEEDS = cpu_count()

cost_model = StructuredMonotoneCost(sizes, npar, mu=MU, Aq='constant', potential=None)
learner = GameLearner(cost_model, CostSampleLoss(), rho=RHO)

print(f"\nFitting StructuredMonotoneCost surrogate "
      f"({N_SEEDS} parallel seeds, {ADAM_EPOCHS} Adam + {LBFGS_EPOCHS} L-BFGS epochs)...")
t0 = time.time()
result = learner.fit(
    (X_train, P_train, J_train), adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
    seeds=np.arange(SEED, SEED + N_SEEDS), val_data=(X_val, P_val, J_val))
surrogate_fit_time = time.time() - t0
print(f"Surrogate fit completed in {surrogate_fit_time:.2f} s")

theta_quad = result.theta[:cost_model.n_param_quad]

# =====================================================================
# Part 2a: extract the learned agents' quadratic cost matrices Q[i], c[i],
# F[i], in the same list-of-N, only-row/entry-block-i-meaningful convention
# as the true game's Q_true, c_true0, F (see generate_random, qp_gnep):
#   J_i(x,p) = 0.5 x^T Q[i] x + c[i](p)^T x_i,  c[i](p) = c[i] + F[i] @ p,
# restricted to row/entry-block i. The surrogate is a single joint
# quadratic game (Aq='constant'): the same (A_hat, q_hat(p)) pair is shared
# by every agent entry below, only the row block i1_p[i]:i2_p[i] differs
# per agent -- mirroring how qp_gnep and solve_surrogate_gne use Q_true/
# c_true0/F and Q=[A_hat]*N, c=[q_hat]*N respectively.
# =====================================================================

i1_p = np.concatenate(([0], np.cumsum(sizes)[:-1]))
i2_p = np.cumsum(sizes)

A_hat0, q_hat0 = cost_model.pseudogradient_matrix(jnp.zeros(npar), theta_quad)
A_hat0 = np.asarray(A_hat0)   # Aq='constant' -> quadratic part is p-independent
q_hat0 = np.asarray(q_hat0)   # q_hat(p=0) = q2 (the constant term)
F_hat_cols = []
for j in range(npar):
    _, q_hat_ej = cost_model.pseudogradient_matrix(jnp.eye(npar)[j], theta_quad)
    F_hat_cols.append(np.asarray(q_hat_ej) - q_hat0)
F_hat = np.column_stack(F_hat_cols)  # (nvar, npar); q_hat(p) = F_hat @ p + q_hat0 (exact, q affine in p)

Q_learned = [A_hat0.copy() for _ in range(N)]
c_learned = [q_hat0.copy() for _ in range(N)]
F_learned = [F_hat.copy() for _ in range(N)]

print("\nLearned parametric quadratic game (Q_learned[i], c_learned[i], "
      "F_learned[i]) vs. the true game's own row/entry block i:")
for i in range(N):
    s, e = i1_p[i], i2_p[i]
    errQ = np.linalg.norm(Q_learned[i][s:e, :] - Q_true[i][s:e, :]) / np.linalg.norm(Q_true[i][s:e, :])
    errc = np.linalg.norm(c_learned[i][s:e] - c_true0[i][s:e]) / np.linalg.norm(c_true0[i][s:e])
    errF = np.linalg.norm(F_learned[i][s:e, :] - F[i][s:e, :]) / np.linalg.norm(F[i][s:e, :])
    print(f"  agent {i + 1}: rel. err Q = {errQ:.4e}, c = {errc:.4e}, F = {errF:.4e}")

# =====================================================================
# Part 2b: R2 of each method's own cost/value fit on the held-out
# validation set (X_val, P_val). Both fits are scored against their own
# native Stage-1 target (see module docstring), since mpfit's value net and
# the surrogate's cost model approximate different quantities.
# =====================================================================

J_val_sur = np.asarray(learner.predict(result, X_val, P_val))  # raw-cost prediction
_, r2_sur_pct, _ = compute_scores(None, None, J_val, J_val_sur, fit='r2')
r2_sur = float(np.mean(r2_sur_pct)) / 100.

J_val_mpfit = np.empty((N_val, N))
for i in range(N):
    XP_i = jnp.hstack((jnp.asarray(X_val)[:, mp.not_i[i]], jnp.asarray(P_val)))
    J_val_mpfit[:, i] = np.asarray(mp._Jhat_batch_fns[i](XP_i, mp.th1[i])).reshape(-1)
_, r2_mpfit_pct, _ = compute_scores(None, None, J_val_br, J_val_mpfit, fit='r2')
r2_mpfit = float(np.mean(r2_mpfit_pct)) / 100.

print(f"\nMean R2 of cost/value fit on the held-out validation set "
      f"({N_val} samples, native target per method):")
print(f"  mpfit value net Jhat_i(x_-i,p) vs best-response value: R2 = {r2_mpfit:.4f}")
print(f"  surrogate cost model J_i(x,p) vs true cost:            R2 = {r2_sur:.4f}")

# =====================================================================
# Part 3: solve the ground-truth LQ-GNEP and the surrogate LQ-GNEP at each
# test parameter with the 'daqp' method (qp_gnep, proximal + full KKT QP),
# reusing the TRUE shared constraints (A_true, b_true(p), E_true, h_true)
# in both cases -- only the quadratic game data (Q, c) differs.
# =====================================================================

QP_GNEP_OPTS = dict(proximal=True, solver='daqp', reduced=False, guler=True,
                     rho=1.e-4, tol=1.e-6, maxiter=10000)


def solve_true_gne(p):
    c_p = [c_true0[i] + F[i] @ p for i in range(N)]
    b_p = b_true0 + S_ineq @ p
    out = qp_gnep(dim=sizes, Q=Q_true, c=c_p, A=A_true, b=b_p,
                  lb=None, ub=None, Aeq=E_true, beq=h_true, **QP_GNEP_OPTS)
    return out.x


def solve_surrogate_gne(p):
    A_hat, q_hat = cost_model.pseudogradient_matrix(jnp.asarray(p), theta_quad)
    A_hat, q_hat = np.asarray(A_hat), np.asarray(q_hat)
    b_p = b_true0 + S_ineq @ p
    out = qp_gnep(dim=sizes, Q=[A_hat] * N, c=[q_hat] * N, A=A_true, b=b_p,
                  lb=None, ub=None, Aeq=E_true, beq=h_true, **QP_GNEP_OPTS)
    return out.x


N_test_s = len(P_test)
X_sur = np.empty((N_test_s, nvar))

if COMPARE_GNE_GROUND_TRUTH:
    X_true = np.empty((N_test_s, nvar))
    t0 = time.time()
    for k in range(N_test_s):
        X_true[k] = solve_true_gne(P_test[k])
    t_true = time.time() - t0

t0 = time.time()
for k in range(N_test_s):
    X_sur[k] = solve_surrogate_gne(P_test[k])
t_sur = time.time() - t0

if COMPARE_GNE_GROUND_TRUTH:
    print(f"\nGround-truth GNE solved via daqp for {N_test_s} test points in {t_true:.2f} s")
print(f"Surrogate GNE solved via daqp for {N_test_s} test points in {t_sur:.2f} s")

# =====================================================================
# Part 4: comparison -- best-response error, constraint violation (both via
# mpfit's own evaluate(), so all three solutions are scored identically),
# and GNE error against the ground-truth solution X_true.
# =====================================================================


def evaluate_solution(X, P, name):
    """Best-response error and constraint violation of X(P), via mpfit's own
    self.A/b_lin/S/lb/ub and self.f (same definitions as res.stats)."""
    N_t = len(P)
    rho_eps = mp._rho_eps
    is_soft = np.isfinite(rho_eps)
    X_br = np.empty_like(X)
    Feas = np.empty((N_t, N))
    J = np.empty((N_t, N))
    for k in range(N_t):
        x_br = np.array(X[k], copy=True)
        for i in range(N):
            xi, flag, _ = mp.best_response_fun(i, X[k], P[k], rho_eps=rho_eps)
            x_br[mp.i1[i]:mp.i2[i]] = xi[:-1] if is_soft else xi
            Feas[k, i] = flag
        J[k] = [mp.f[i](X[k], P[k]) for i in range(N)]
        X_br[k] = x_br
    stats = mp.evaluate(P, X, Feas, X_br, J, split_name=name, verbose=False)
    # mp.evaluate() only exposes the reduced E[||xhat-BR(xhat)||^2] (mean of
    # SQUARED per-sample norms); recompute from X/X_br directly so this
    # reports mean_k ||xhat_k - BR(xhat_k)||_2 (mean of UNSQUARED norms),
    # matching the convention used by example_quad_game.py/
    # example_nl_internet.py/example_cstr_control.py and the definition
    # stated in the paper (Sec. 5, "Both the BR and NE errors...").
    Feas_bool = np.prod(Feas, axis=1) > 0
    stats.br_err_mean_feasible = float(
        np.mean(np.linalg.norm(X - X_br, axis=1)[Feas_bool]))
    return stats


stats_mpfit = evaluate_solution(X_mpfit, P_test, "mpfit (explicit NN)")
stats_sur = evaluate_solution(X_sur, P_test, "surrogate LQ-GNE (daqp)")

if COMPARE_GNE_GROUND_TRUTH:
    stats_true = evaluate_solution(X_true, P_test, "ground truth (daqp)")
    gne_err_mpfit = np.linalg.norm(X_mpfit - X_true, axis=1)
    gne_err_sur = np.linalg.norm(X_sur - X_true, axis=1)

# =====================================================================
# Part 4b: does the surrogate's PSEUDOGRADIENT (not just its cost VALUES)
# match the true one? A high R2 on J_i(x,p) does not certify a small
# residual on the OPERATOR that actually defines the vGNE. Cross-check
# against the strongly-monotone-VI sensitivity bound: for two vGNEs
# x1=vGNE(F1,C), x2=vGNE(F2,C) on the same feasible set C, with F1
# mu-strongly monotone,
#     ||x1 - x2|| <= ||F1(x2) - F2(x2)|| / mu
# (Facchinei-Pang Prop. 12.11-type argument; no active-set discontinuity).
# Take F1=true pseudogradient (mu=0.1, certified by generate_random),
# F2=surrogate pseudogradient, x2=X_sur. F1, F2 are the raw (generally
# asymmetric) assembled matrices -- what qp_gnep's stationarity condition
# actually uses (only the QP's Hessian is symmetrized internally, not the
# operator), so M_true/A_hat0 stay unsymmetrized here for a fair comparison.
# Reuses Q_learned/c_learned/F_learned/A_hat0/i1_p/i2_p from Part 2a; only
# meaningful when COMPARE_GNE_GROUND_TRUTH computed gne_err_sur above.
# =====================================================================

if COMPARE_GNE_GROUND_TRUTH:
    M_true = np.zeros((nvar, nvar))
    for i in range(N):
        M_true[i1_p[i]:i2_p[i], :] = Q_true[i][i1_p[i]:i2_p[i], :]
    rel_err_A = np.linalg.norm(A_hat0 - M_true) / np.linalg.norm(M_true)

    mismatch_norms = np.empty(N_test_s)
    for k in range(N_test_s):
        p = P_test[k]
        f_true_p = np.concatenate([(c_true0[i] + F[i] @ p)[i1_p[i]:i2_p[i]] for i in range(N)])
        q_hat_p = F_hat @ p + q_hat0
        F_true_at_xsur = M_true @ X_sur[k] + f_true_p
        F_sur_at_xsur = A_hat0 @ X_sur[k] + q_hat_p
        mismatch_norms[k] = np.linalg.norm(F_true_at_xsur - F_sur_at_xsur)

    bound_rhs = mismatch_norms / mu  # mu = 0.1, the TRUE game's certified strong-monotonicity constant

    print("\n" + "=" * 78)
    print("Diagnostic: pseudogradient (operator) mismatch vs. cost-value R2")
    print("=" * 78)
    print(f"Quadratic part relative error  ||A_hat - Q_true|| / ||Q_true|| (raw, "
          f"unsymmetrized) = {rel_err_A:.4e}")
    print(f"Operator mismatch ||F_true(X_sur)-F_sur(X_sur)|| (mean/max)  = "
          f"{mismatch_norms.mean():.4e} / {mismatch_norms.max():.4e}")
    print(f"VI sensitivity bound  ||mismatch||/mu             (mean/max) = "
          f"{bound_rhs.mean():.4e} / {bound_rhs.max():.4e}")
    print(f"Observed GNE error vs ground truth                (mean/max) = "
          f"{gne_err_sur.mean():.4e} / {gne_err_sur.max():.4e}")
    print("=" * 78)
else:
    print("\nSkipping pseudogradient-mismatch diagnostic "
          "(COMPARE_GNE_GROUND_TRUTH = False -- no ground-truth GNE to compare against).")

print("\n" + "=" * 78)
print("Comparison on the common test set "
      f"({N_test_s} samples): mpfit (explicit NN) vs. StructuredMonotoneCost "
      "surrogate (daqp)")
print("=" * 78)
header = f"{'':32s}{'mpfit (NN)':>18s}{'surrogate (daqp)':>22s}"
print(header)
print(f"{'BR error':32s}"
      f"{stats_mpfit.br_err_mean_feasible:18.4e}{stats_sur.br_err_mean_feasible:22.4e}")
print(f"{'constraint violation (mean)':32s}"
      f"{stats_mpfit.viol_mean:18.4e}{stats_sur.viol_mean:22.4e}")
print(f"{'constraint violation (max)':32s}"
      f"{stats_mpfit.viol_max:18.4e}{stats_sur.viol_max:22.4e}")
if COMPARE_GNE_GROUND_TRUTH:
    print(f"{'GNE error vs ground truth (mean)':32s}"
          f"{gne_err_mpfit.mean():18.4e}{gne_err_sur.mean():22.4e}")
    print(f"{'GNE error vs ground truth (max)':32s}"
          f"{gne_err_mpfit.max():18.4e}{gne_err_sur.max():22.4e}")
print(f"{'cost/value fit R2 (mean, val.)':32s}"
      f"{r2_mpfit:18.4f}{r2_sur:22.4f}")
print(f"{'fitting time (s)':32s}"
      f"{res.training_time:18.2f}{surrogate_fit_time:22.2f}")
print(f"{'per-sample solve time (s)':32s}"
      f"{res.t_predict / N_test_s:18.2e}{t_sur / N_test_s:22.2e}")
print("=" * 78)
if COMPARE_GNE_GROUND_TRUTH:
    print("Ground-truth solve sanity check (daqp on the true game): "
          f"best-response error (mean) = {stats_true.br_err_mean_feasible:.4e}, "
          f"constraint violation (max) = {stats_true.viol_max:.4e}")

# =====================================================================
# Part 5: print the comparison table above as LaTeX (no file written).
# =====================================================================


def sci_to_latex(value, decimals=5, bold=False):
    """Format a float in decimal notation with a fixed number of decimal digits,
    e.g. 0.0000496 -> $0.00005$ (or $\\mathbf{0.00005}$ if bold=True)."""
    body = f"{value:.{decimals}f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"$ {body} $"


def exp_to_latex(value, decimals=4, bold=False):
    """Format a float in exponential notation with a fixed number of mantissa
    digits, e.g. 0.0000496 -> $4.9600 \\cdot 10^{-5}$ (or bold if bold=True)."""
    exp = 0 if value == 0 else int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exp)
    body = f"{mantissa:.{decimals}f} \\cdot 10^{{{exp}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"$ {body} $"


rows = [
    # (name, mpfit value, surrogate value, decimals, higher_is_better, exponential)
    ("best-response error (mean)", stats_mpfit.br_err_mean_feasible, stats_sur.br_err_mean_feasible, 4, False, True),
    ("constraint violation (mean)", stats_mpfit.viol_mean, stats_sur.viol_mean, 4, False, True),
    ("constraint violation (max)", stats_mpfit.viol_max, stats_sur.viol_max, 4, False, True),
]
if COMPARE_GNE_GROUND_TRUTH:
    rows += [
        ("GNE error vs ground truth (mean)", gne_err_mpfit.mean(), gne_err_sur.mean(), 4, False, False),
        ("GNE error vs ground truth (max)", gne_err_mpfit.max(), gne_err_sur.max(), 4, False, False),
    ]
rows += [
    ("opt.cost/cost fit (average R$^2$ score)", r2_mpfit, r2_sur, 4, True, False),
    ("fitting time (s)", res.training_time, surrogate_fit_time, 2, False, False),
    ("per-sample solve time ($\\mu$s)", 1e6 * res.t_predict / N_test_s, 1e6 * t_sur / N_test_s, 2, False, False),
]

latex_lines = [
    "% Auto-generated by example_surrogate_vs_explicit.py",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Approximate explicit GNE solution~\\cite{BT26} vs. LQ-GNE surrogates" +
    " on a random strongly-monotone LQ-GNEP.}",
    "\\label{tab:surrogate_vs_explicit}",
    "\\setlength{\\tabcolsep}{5pt}",
    "\\renewcommand{\\arraystretch}{1.} % Adjust row separation",
    "\\begin{tabular}{l|r|r}",
    "\\toprule",
    " & explicit~\\cite{BT26} & surrogate GNE \\\\",
    "\\midrule",
]
for name, v_mpfit, v_sur, decimals, higher_is_better, exponential in rows:
    bold_mpfit = v_mpfit >= v_sur if higher_is_better else v_mpfit <= v_sur
    bold_sur = v_sur >= v_mpfit if higher_is_better else v_sur <= v_mpfit
    fmt = exp_to_latex if exponential else sci_to_latex
    latex_lines.append(
        f"{name} & {fmt(v_mpfit, decimals, bold_mpfit)} & "
        f"{fmt(v_sur, decimals, bold_sur)} \\\\")
latex_lines += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]
latex_table = "\n".join(latex_lines) + "\n"

print("\n" + latex_table)

# =====================================================================
# Part 6: number of weights of mpfit's explicit GNE solution network
# p -> x*(p) (the skip-connection MLP returned by mp.solution_network()).
# =====================================================================

n_weights_mpfit = sum(w.size for w in mp.solution_network(print=False))
print(f"Number of weights of mpfit's GNE solution network: {n_weights_mpfit}")
