"""
Compare two ways of getting a fast parametric solution p -> x*(p) to a
random monotone parametric GNEP, whose TRUE cost is specified generically
in JAX (not assumed linear-quadratic), against the ground-truth GNE:

1. mpfit.NL_GNEP: a neural-network explicit solution p -> x*(p), fit on
   best-response data obtained by solving each agent's best response via
   penalty-constrained L-BFGS-B on the raw cost f_i(x,p).
2. StructuredMonotoneCost (Aq='NN'): a monotone-by-construction surrogate
   whose pseudogradient matrix C(p), D(p) and offset q(p) are all emitted
   by a hypernetwork MLP of p, fit on cost samples J_i(x,p) via
   GameLearner/CostSampleLoss. USE_POTENTIAL_IN_SURROGATE optionally adds
   a shared input-convex NN potential Psi(x,p) to every agent's cost,
   which preserves monotonicity but makes the surrogate no longer a
   quadratic game even at a fixed p, so it must then be solved online with
   a generic KKT solve instead of a closed-form QP.

The true cost f_i(x,p) is defined by agent_cost(i), selected by the
COST_TYPE flag below:
  'quadratic'  : a convex quadratic cost, used to validate the pipeline
                 against a case with known closed-form structure.
  'logsumexp'  : a genuinely nonlinear, non-quadratic cost,
                     f_i(x,p) = log sum_r exp(a_ir^T x_i + b_ir^T x_-i
                                               + gamma_ir^T p)
                                + lambda_i(p) * ||x_i/10||_4^4,
                     lambda_i(p) = 1 + p_1^2 + p_2^2, a_ir/b_ir/gamma_ir ~
                     N(0,1) -- convex in x_i, but not quadratic.
Nothing downstream depends on the cost being quadratic; only the Part 2a/4b
diagnostics (which compare the learned surrogate against the true quadratic
matrices) require COST_TYPE == 'quadratic' and are skipped otherwise.

Both are evaluated on the same test parameters p against:

    - best-response error   E[||xhat - BR(xhat)||]  (mean of UNSQUARED
                             per-sample norms, feasible BRs only)
    - constraint violation  mean/max of the box bounds lb <= x <= ub and of
                             the shared inequalities A_true x <= b_true0 +
                             S_true@p
    - GNE error             ||xhat(p) - x_true(p)||, x_true(p) solving the
                             TRUE GNEP via a generic KKT least-squares solve
    - cost/value fit R2     on a held-out validation set, using each
                             method's own Stage-1 target: mpfit's value net
                             Jhat_i(x_{-i},p) fits the best-response VALUE
                             min_{x_i} f_i(x,p); the surrogate's cost model
                             fits the raw cost J_i(x,p) directly

Solving the TRUE GNEP at every test p (Part 3) is an expensive nonlinear
KKT solve; the COMPARE_GNE_GROUND_TRUTH flag makes this optional. When
False, X_true/stats_true/the GNE-error metrics and the Part 4b
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
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from jax_sysid.utils import compute_scores

from jax.scipy.special import logsumexp

from nashopt import GNEP
from nashopt.lq.generate_random import generate_random  # requires nashopt>=1.3.0
from nashopt.lq.qp_gnep import qp_gnep

from mpfit import NL_GNEP
from monotone_games import CostSampleLoss, EquilibriumSolver, GameLearner, StructuredMonotoneCost

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

np.random.seed(0)

# =====================================================================
# User-defined parameters and flags (everything tunable lives here; the
# rest of the file only defines derived quantities and pipeline logic).
# =====================================================================

# --- game size ---
N = 2      # number of agents
n = 2      # number of decision variables per agent
npar = 2   # number of parameters
m = 10      # number of shared inequality constraints A_true x <= b_true0 + S_true@p

# --- cost model flags (see agent_cost(i) in Part 1 for what each selects) ---
COST_TYPE = 'logsumexp'          # 'quadratic' or 'logsumexp'
COMPARE_GNE_GROUND_TRUTH = False  # solve/compare vs the TRUE GNEP via mp.solve_gne
                                  # (Part 3/4/4b) -- an expensive nonlinear
                                  # solve per test point; False skips it and
                                  # only reports best-response error,
                                  # constraint violation, and cost/value fit R2

# --- true-game cost parameters ---
mu = 0.1      # monotone_eps used to build the quadratic test instance (COST_TYPE == 'quadratic')
m_terms = 3   # number of exp-terms in the log-sum-exp cost, per agent (COST_TYPE == 'logsumexp')

# --- dataset sizes ---
N_train = 2000 # number of training samples
N_val = 1000   # number of validation samples
N_test = 1000  # number of test samples

# --- decision-variable / parameter ranges ---
nvar = N * n
sizes = [n] * N
pmin = 0. * np.ones(npar)
pmax = 1. * np.ones(npar)
lb = -2. * np.ones(nvar)
ub = 2. * np.ones(nvar)

# Box used only to sample x in generate_br_data() below, decoupled from the
# actual variable bounds lb/ub (as in example_surrogate_vs_explicit.py):
# useful to narrow/widen the *training-data sampling* range independently of
# the constraint box, e.g. if lb/ub were relaxed to +-inf and the game's
# natural operating range is narrower than the resulting unbounded box.
# Set equal to lb/ub here since they already form a moderate, finite box.
xlb_sample = lb.copy()
xub_sample = ub.copy()

# --- mpfit.NL_GNEP explicit-solution network architecture (Part 1) ---
value_nn = [10, 5]
nn_value_act = jax.nn.swish
n1, n2 = 10, 5
solution_nn_act = jax.nn.relu

# --- StructuredMonotoneCost surrogate fitting (Part 2) ---
MU = 0.0           # strong-monotonicity constant of the surrogate
RHO = 1.e-8        # L2 regularization on the cost-model parameters
AQ_NN_LAYERS = [5, 5]   # hidden widths of the C(p),D(p),q(p) hypernetwork (Aq='NN')
AQ_NN_ACT = jax.nn.relu  # activation of the C(p),D(p),q(p) hypernetwork
USE_POTENTIAL_IN_SURROGATE = True  # add a shared input-convex NN potential
                                    # Psi(x,p) to the surrogate's cost (see
                                    # StructuredMonotoneCost.potential).
                                    # False: potential=None, pure quadratic
                                    # surrogate solved via the closed-form
                                    # EquilibriumSolver.solve_quadratic.
                                    # True: potential='NN', more capacity to
                                    # track a nonlinear true game, but needs
                                    # the generic EquilibriumSolver.solve
                                    # nonlinear KKT solve instead.
POTENTIAL_NN_LAYERS = [5, 5]      # hidden widths of the shared input-convex
                                  # potential Psi(x,p) network (potential='NN')
POTENTIAL_NN_ACT = jax.nn.softplus  # convex, nondecreasing activation
                                     # (required for input-convexity)
ADAM_EPOCHS = 1000
LBFGS_EPOCHS = 5000
SEED = 3
N_SEEDS = cpu_count()

# --- surrogate GNE solve options (EquilibriumSolver, Part 3) ---
SURROGATE_SOLVE_OPTS = {'solver': 'hybr'}
USE_EXTRAGRAD = True  # if True, solve the surrogate GNE at each test p via
                       # nashopt.GNEP.solve(solver='extragrad') (Korpelevich's
                       # extragradient method for the variational GNE) instead
                       # of the KKT root-find/closed-form QP path below,
                       # regardless of USE_POTENTIAL_IN_SURROGATE. The step
                       # size is set to alpha = 0.99/L(p), with L(p) =
                       # cost_model.lipschitz_constant(result.theta, p) an
                       # estimate/bound on the Lipschitz constant of the
                       # learned pseudogradient at that p (paper's
                       # Proposition 4.2), evaluated once per test p.
extragrad_max_iter = 50  # max iterations for extragrad_nlgnep (only used when USE_EXTRAGRAD)

# Log-sum-exp cost coefficients (only used when COST_TYPE == 'logsumexp'):
# a_ir in R^n (agent i's own block), b_ir in R^(nvar-n) (all other agents'
# blocks x_-i), gamma_ir in R^npar, r = 1,...,m_terms -- drawn uniformly in
# [-0.9, 0.9] and rounded to one decimal digit.
rng_cost = np.random.default_rng(1)
A_ls = [np.round(rng_cost.uniform(-.9, .9, (m_terms, n)), 1) for _ in range(N)]
B_ls = [np.round(rng_cost.uniform(-.9, .9, (m_terms, nvar - n)), 1) for _ in range(N)]
G_ls = [np.round(rng_cost.uniform(-.9, .9, (m_terms, npar)), 1) for _ in range(N)]

# =====================================================================
# Part 1: true game with generic (initially quadratic) agent costs, and
# mpfit's explicit solution p -> x*(p) via NL_GNEP.
# =====================================================================

# Random monotone quadratic data (m shared inequality constraints A x <= b,
# q=0: no shared equalities, on top of the box bounds lb/ub above), used to
# instantiate agent_cost(i) when COST_TYPE == 'quadratic' (and, regardless
# of COST_TYPE, as the reference operator M_true/Q_true for the
# quadratic-only diagnostics in Parts 2a/4b, and as the TRUE game's shared
# constraint set A_true x <= b_true0 + S_true@p for every COST_TYPE).
gnep_true, data = generate_random(dim=sizes, m=m, m_act=0, q=0, seed=0, mu=mu)
Q_true = gnep_true.Q  # list of N matrices, shape (nvar, nvar), only row-block i used
c_true0 = gnep_true.c  # list of N vectors, shape (nvar,), only entries [i*n:(i+1)*n] used
A_true = gnep_true.A  # shape (m, nvar); shared constraint A_true x <= b_true0 + S_true@p
b_true0 = gnep_true.b  # shape (m,)
F = [np.round(2.0 * np.random.rand(nvar, npar), 2) for _ in range(N)]
S_true = np.round(1.0 * np.random.rand(m, npar), 2)  # parametrizes the shared
# inequality RHS: A_true x <= b_true0 + S_true @ p (S_true >= 0, shape (m, npar))

G_true = data["G"]  # true pseudogradient, generally asymmetric
mu_check = np.linalg.eigvalsh(0.5 * (G_true + G_true.T)).min()
print(f"Monotonicity check (generate_random): "
      f"lambda_min(0.5*(G+G.T)) = {mu_check:.4e}")

def agent_cost(i):
    """Return J_i(x, p) for agent i, as a JAX-jittable function fi(x, p),
    selected by COST_TYPE (see module docstring)."""
    i1, i2 = i * n, (i + 1) * n

    if COST_TYPE == 'quadratic':
        Qi = jnp.asarray(Q_true[i])
        ci0 = jnp.asarray(c_true0[i])
        Fi = jnp.asarray(F[i])

        @jax.jit
        def fi(x, p):
            c_i = ci0 + Fi @ p
            return 0.5 * x @ (Qi @ x) + c_i @ x

        return fi

    elif COST_TYPE == 'logsumexp':
        Ai = jnp.asarray(A_ls[i])
        Bi = jnp.asarray(B_ls[i])
        Gi = jnp.asarray(G_ls[i])

        @jax.jit
        def fi(x, p):
            xi = x[i1:i2]
            x_mi = jnp.concatenate((x[:i1], x[i2:]))
            z = Ai @ xi + Bi @ x_mi + Gi @ p
            lam = 1. + p[0] + p[1]
            return logsumexp(z) + lam * jnp.sum((xi / 10.) ** 4)

        return fi

    else:
        raise ValueError(f"Unknown COST_TYPE: {COST_TYPE!r}")


f = [agent_cost(i) for i in range(N)]

# Generic best-response solver for the TRUE game: nashopt.GNEP solves each
# agent's subproblem min_{x_i} f_i(x,p) s.t. lb_i<=x_i<=ub_i,
# A_true x<=b_true0+S_true@p via penalty-constrained L-BFGS-B (jaxopt), using
# only the JAX-autodiff gradient of fi -- no closed form or quadratic
# structure is exploited here. The shared constraint is passed through GNEP's
# g(x,p)<=0 interface (g must take (x,p) since parametric=True); ng=m is 0
# (g unused) when m==0.
g_true = (lambda x, p: A_true @ x - (b_true0 + S_true @ p)) if m > 0 else None
gnep_true = GNEP(sizes=sizes, f=f, g=g_true, ng=m, lb=lb, ub=ub, parametric=True, npar=npar)


def best_response_fun(i, x, p):
    """Generic BR solver required by NL_GNEP: (i, x, p) -> (xi, flag, Ji)."""
    sol = gnep_true.best_response(i, jnp.asarray(x), p=jnp.asarray(p))
    i1, i2 = gnep_true.i1[i], gnep_true.i2[i]
    xi = np.asarray(sol.x[i1:i2])
    x_full = np.asarray(sol.x)
    # Feasibility tolerance is loosened to 1e-4 (not 1e-6): best_response()
    # enforces A_true x <= b_true0 + S_true@p only via an L-BFGS-B penalty
    # (rho=1e5 default), leaving a residual violation ~1/rho ~ 1e-5 even at
    # an exact best response. A 1e-6 tolerance would spuriously flag every
    # best response as infeasible when x already sits on an active
    # constraint (as a converged GNE typically does), collapsing
    # br_err_mean_feasible to nan (mean of an empty selection).
    feasible = bool(np.all(xi >= lb[i1:i2] - 1e-4) and np.all(xi <= ub[i1:i2] + 1e-4))
    if m > 0:
        feasible = feasible and bool(np.all(A_true @ x_full - (b_true0 + S_true @ p) <= 1e-4))
    return xi, feasible, float(sol.f)


def generate_br_data(N_data, seed=None):
    """Sample (x,p) uniformly in the box/parameter ranges, then compute each
    agent's best response in parallel to build the (X, P, J) training data
    NL_GNEP needs (J = best-response VALUE, the Stage-1 fitting target).
    x is sampled from xlb_sample/xub_sample (not lb/ub directly), decoupling
    the training-data sampling box from the actual variable bounds."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(pmin, pmax, size=(N_data, npar))
    X = rng.uniform(xlb_sample, xub_sample, size=(N_data, nvar))

    pairs = [(i, k) for i in range(N) for k in range(N_data)]

    def _br(i, k):
        _, _, Ji = best_response_fun(i, X[k], P[k])
        return Ji

    results = Parallel(n_jobs=-1)(
        delayed(_br)(i, k)
        for i, k in tqdm(pairs, desc="BR data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    )
    J = np.empty((N_data, N))
    for idx, (i, k) in enumerate(pairs):
        J[k, i] = results[idx]
    return X, P, J


mp = NL_GNEP(N, sizes, npar, f, best_response_fun, generate_br_data,
             A=A_true if m > 0 else None, b=b_true0 if m > 0 else None,
             S=S_true if m > 0 else None,
             lb=lb, ub=ub, pmin=pmin, pmax=pmax,
             n1=n1, n2=n2, value_nn=value_nn, nn_value_act=nn_value_act,
             solution_nn_act=solution_nn_act, beta=10)
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

potential_spec = ({'type': 'NN', 'layers': POTENTIAL_NN_LAYERS, 'activation': POTENTIAL_NN_ACT}
                   if USE_POTENTIAL_IN_SURROGATE else None)
cost_model = StructuredMonotoneCost(
    sizes, npar, mu=MU, potential=potential_spec,
    Aq={'type': 'NN', 'layers': AQ_NN_LAYERS, 'activation': AQ_NN_ACT})
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
# Part 2a: extract a p=0 SNAPSHOT of the learned agents' quadratic cost
# matrices Q[i], c[i], F[i], in the same list-of-N, only-row/entry-block-i-
# meaningful convention as the true (currently quadratic) game's Q_true,
# c_true0, F:
#     J_i(x,p) = 0.5 x^T Q[i] x + c[i](p)^T x_i,
#     c[i](p) = c[i] + F[i] @ p, restricted to row/entry-block i.
# Only meaningful while agent_cost(i) returns a quadratic function -- it no
# longer applies once agent_cost is nonlinear (the surrogate would then only
# match locally). A_hat(p), q_hat(p) cover only the surrogate's separable
# quadratic part (pseudogradient_matrix, Aq='NN'): when
# USE_POTENTIAL_IN_SURROGATE is True they exclude the potential term's
# x-dependent contribution grad_x Psi(x,p;theta_pot), which cannot be
# folded into a constant Q_learned[i]/c_learned[i] snapshot at all. Aq='NN'
# also makes A_hat0/q_hat0 only a p=0 evaluation (no longer p-independent/
# affine as under Aq='constant'), with F_hat only the finite-difference
# slope of q_hat(.) at p=0, not an exact affine coefficient. So even for
# COST_TYPE == 'quadratic' this is a rough, local (p=0, quadratic-part-only)
# check, not exact recovery; Part 4b's operator-mismatch diagnostic accounts
# for the potential term too.
# =====================================================================

i1_p = np.concatenate(([0], np.cumsum(sizes)[:-1]))
i2_p = np.cumsum(sizes)

A_hat0, q_hat0 = cost_model.pseudogradient_matrix(jnp.zeros(npar), theta_quad)
A_hat0 = np.asarray(A_hat0)   # A_hat(p=0) snapshot (Aq='NN' -> generally p-dependent)
q_hat0 = np.asarray(q_hat0)   # q_hat(p=0) snapshot
F_hat_cols = []
for j in range(npar):
    _, q_hat_ej = cost_model.pseudogradient_matrix(jnp.eye(npar)[j], theta_quad)
    F_hat_cols.append(np.asarray(q_hat_ej) - q_hat0)
F_hat = np.column_stack(F_hat_cols)  # (nvar, npar); finite-diff slope of q_hat(.) at p=0
# (exact only if the fitted q_hat happens to be affine in p; Aq='NN' does not guarantee this)

Q_learned = [A_hat0.copy() for _ in range(N)]
c_learned = [q_hat0.copy() for _ in range(N)]
F_learned = [F_hat.copy() for _ in range(N)]

if COST_TYPE == 'quadratic':
    print("\nLearned parametric quadratic game p=0 snapshot (Q_learned[i], "
          "c_learned[i], F_learned[i]) vs. the true game's own row/entry block i:")
    for i in range(N):
        s, e = i1_p[i], i2_p[i]
        errQ = np.linalg.norm(Q_learned[i][s:e, :] - Q_true[i][s:e, :]) / np.linalg.norm(Q_true[i][s:e, :])
        errc = np.linalg.norm(c_learned[i][s:e] - c_true0[i][s:e]) / np.linalg.norm(c_true0[i][s:e])
        errF = np.linalg.norm(F_learned[i][s:e, :] - F[i][s:e, :]) / np.linalg.norm(F[i][s:e, :])
        print(f"  agent {i + 1}: rel. err Q = {errQ:.4e}, c = {errc:.4e}, F = {errF:.4e}")
else:
    print(f"\nSkipping learned-vs-true quadratic-game comparison "
          f"(COST_TYPE = {COST_TYPE!r}, true cost is not quadratic).")

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
# Part 2c: fit a SECOND mpfit.NL_GNEP explicit-solution network, xhat(p),
# with the SAME architecture as mp (Part 1: n1, n2, value_nn, nn_value_act,
# solution_nn_act), but approximating the SURROGATE game's OWN solution
# p -> x*(p) instead of the true game's. Best-response training data are
# generated from the surrogate's cost model (cost_model, with theta fixed
# to result.theta from Part 2) via a fresh nashopt.GNEP (gnep_sur) built on
# the surrogate's per-agent costs f_sur[i](x,p) = cost_model.costs(x,p,
# theta_sur)[i], subject to the same box/shared constraints (lb, ub, A_true,
# b_true0, S_true) used for the surrogate GNE solve in Part 3 -- mirroring
# exactly how gnep_true/best_response_fun/generate_br_data (Part 1) generate
# best-response data for the true game, just with f_sur/gnep_sur in place of
# f/gnep_true. xhat(P_test) is then used in Part 3 below to warm-start the
# surrogate GNE solve, replacing the previous warm start X_mpfit[k] (the
# TRUE game's own explicit solution) with a warm start that approximates
# the SURROGATE's own equilibrium map -- a better-matched initial guess for
# solving the surrogate GNEP.
# =====================================================================

theta_sur = result.theta  # surrogate cost-model parameters, fixed from Part 2


def make_f_sur(i):
    @jax.jit
    def fi(x, p):
        return cost_model.costs(x, p, theta_sur)[i]
    return fi


f_sur = [make_f_sur(i) for i in range(N)]

g_sur = (lambda x, p: A_true @ x - (b_true0 + S_true @ p)) if m > 0 else None
gnep_sur = GNEP(sizes=sizes, f=f_sur, g=g_sur, ng=m, lb=lb, ub=ub, parametric=True, npar=npar)


def best_response_fun_sur(i, x, p):
    """Best response of agent i under the SURROGATE cost model: generic BR
    solver required by NL_GNEP, mirrors best_response_fun (Part 1) but
    against gnep_sur/f_sur instead of gnep_true/f."""
    sol = gnep_sur.best_response(i, jnp.asarray(x), p=jnp.asarray(p))
    i1, i2 = gnep_sur.i1[i], gnep_sur.i2[i]
    xi = np.asarray(sol.x[i1:i2])
    x_full = np.asarray(sol.x)
    feasible = bool(np.all(xi >= lb[i1:i2] - 1e-4) and np.all(xi <= ub[i1:i2] + 1e-4))
    if m > 0:
        feasible = feasible and bool(np.all(A_true @ x_full - (b_true0 + S_true @ p) <= 1e-4))
    return xi, feasible, float(sol.f)


def generate_br_data_sur(N_data, seed=None):
    """Same sampling scheme as generate_br_data (Part 1), but against the
    surrogate's own best response (best_response_fun_sur)."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(pmin, pmax, size=(N_data, npar))
    X = rng.uniform(xlb_sample, xub_sample, size=(N_data, nvar))

    pairs = [(i, k) for i in range(N) for k in range(N_data)]

    def _br(i, k):
        _, _, Ji = best_response_fun_sur(i, X[k], P[k])
        return Ji

    results = Parallel(n_jobs=-1)(
        delayed(_br)(i, k)
        for i, k in tqdm(pairs, desc="BR data (surrogate)", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    )
    J = np.empty((N_data, N))
    for idx, (i, k) in enumerate(pairs):
        J[k, i] = results[idx]
    return X, P, J


print("\nFitting explicit-solution network xhat(p) for the SURROGATE game "
      "(same architecture as Part 1's mp, best-response data generated from "
      "the surrogate's own cost model)...")
mp_sur = NL_GNEP(N, sizes, npar, f_sur, best_response_fun_sur, generate_br_data_sur,
                 A=A_true if m > 0 else None, b=b_true0 if m > 0 else None,
                 S=S_true if m > 0 else None,
                 lb=lb, ub=ub, pmin=pmin, pmax=pmax,
                 n1=n1, n2=n2, value_nn=value_nn, nn_value_act=nn_value_act,
                 solution_nn_act=solution_nn_act, beta=10)
mp_sur.setup(N_train=N_train, N_val=N_val)
mp_sur.solve()

# xhat's own fitting time is NOT folded into surrogate_fit_time: the
# reported "fitting time" row is meant to report only the cost of
# constructing the surrogate cost model itself (Part 2's learner.fit()),
# not the cost of fitting xhat -- xhat's cost is instead amortized into the
# per-test-point solve time below (t_predict_sur, alongside the extragradient
# iterations), since both are part of evaluating the surrogate GNE at test
# time rather than part of building the surrogate.
print(f"xhat(p) fitting completed in {mp_sur.training_time:.2f} s")

# Evaluate xhat(p) at the SAME test parameters P_test used throughout (Part
# 1's res.P_test, generated by mp.test() with its fixed seed=42): this is
# the warm start used for the surrogate GNE solve in Part 3.
t0 = time.time()
X_hat_sur = mp_sur.predict(P_test)
t_predict_sur = time.time() - t0

# =====================================================================
# Part 3: solve the ground-truth GNEP and the surrogate GNEP at each test
# parameter.
#   - Ground truth: mp.solve_gne(p) -- generic, builds a nashopt.GNEP with p
#     baked into the TRUE (possibly nonlinear) costs f_i and box bounds, and
#     solves its KKT system via least-squares. Valid regardless of whether
#     agent_cost(i) is quadratic.
#   - Surrogate: with USE_POTENTIAL_IN_SURROGATE False (potential=None), the
#     game is quadratic at every fixed p, so EquilibriumSolver.solve_quadratic
#     solves the NE directly (or via qp_gnep's closed-form KKT-QP with box
#     bounds). With it True, the potential's grad_x Psi(x,p) makes the
#     pseudogradient nonlinear in x even at fixed p, so EquilibriumSolver.
#     solve is used instead -- the same generic nashopt KKT machinery as
#     mp.solve_gne, applied to cost_model.costs(x,p,theta) with the full
#     theta (quadratic part + potential).
# =====================================================================

# EquilibriumSolver.solve_quadratic (qp_gnep/direct linear solve, used when
# USE_POTENTIAL_IN_SURROGATE is False) reads 'A'/'b'; solve() (used when it
# is True) reads 'A'/'b'/'S' and builds the p-dependent shared constraint
# A_true@x <= b_true0 + S_true@p internally -- both are set below so the
# same surrogate_eq instance works regardless of USE_POTENTIAL_IN_SURROGATE,
# with the RHS's p-dependence captured once (either by qp_gnep's per-call
# 'b', cheap since that path is already a closed-form QP solve, or by
# solve()'s one-time JIT trace, auto-compiled on its first call below)
# rather than by refreshing 'b'/'g' inside solve_surrogate_gne() on every
# test point.
surrogate_constraints = {'lb': lb, 'ub': ub}
if m > 0:
    surrogate_constraints['A'] = A_true
    surrogate_constraints['b'] = b_true0
    surrogate_constraints['S'] = S_true
surrogate_eq = EquilibriumSolver(cost_model, constraints=surrogate_constraints)
if USE_EXTRAGRAD:
    surrogate_desc = "surrogate GNE (extragradient)"
elif USE_POTENTIAL_IN_SURROGATE:
    surrogate_desc = "surrogate GNE (nashopt KKT)"
else:
    surrogate_desc = "surrogate GNE (quadratic KKT)"

if USE_EXTRAGRAD:
    # Compile the parametric surrogate's pseudogradient F(x,p;theta) and
    # shared-constraint Jacobian once, right after training (result.theta is
    # fixed from here on), with p a genuine JAX-traced argument, and build
    # the IPOPT/TRF projectors once (EquilibriumSolver.
    # compile_parametric_extragrad) -- so solve_extragrad() below reuses
    # these for every test point instead of rebuilding a fresh
    # nashopt.GNEP (and hence fresh jax.jit wrappers and IPOPT problem) per
    # call, which is what nashopt.GNEP.solve(solver='extragrad') on a
    # one-off GNEP per test point would do. This explicit call is optional
    # -- solve_extragrad() would compile automatically on its first call
    # regardless -- it is only here to report compile time separately.
    t0 = time.time()
    surrogate_eq.compile_parametric_extragrad(result.theta)
    print(f"Parametric surrogate extragradient operator compiled in {time.time() - t0:.2f} s")
elif USE_POTENTIAL_IN_SURROGATE:
    # Compile the parametric surrogate GNEP once, right after training
    # (result.theta is fixed from here on): the KKT residual/Jacobian are
    # JIT-compiled as functions of p treated as a genuine JAX-traced
    # argument, so solve() below reuses that single compiled executable for
    # every test point (since it is called with the SAME result.theta
    # object throughout) instead of retracing/recompiling per call (the
    # actual cause of surrogate_eq.solve() being slow in a loop over many
    # test points before this library-level fix). This explicit call is
    # optional -- solve() would compile automatically on its first call
    # regardless -- it is only here to report compile time separately.
    t0 = time.time()
    surrogate_eq.compile_parametric(result.theta)
    print(f"Parametric surrogate GNEP compiled in {time.time() - t0:.2f} s")


def solve_surrogate_gne(p, x0=None):
    if USE_EXTRAGRAD:
        # Estimate the Lipschitz constant of the learned pseudogradient at
        # this p (Proposition 4.2) and set the extragradient step size to
        # 0.99/L, then solve the (monotone-by-construction) surrogate's
        # variational GNE via EquilibriumSolver.solve_extragrad()
        # (Korpelevich's extragradient method, mirroring nashopt.GNEP.
        # solve(solver='extragrad') but reusing the once-compiled F(x,p;theta)
        # and projectors from compile_parametric_extragrad() above instead of
        # rebuilding a fresh GNEP per test point).
        theta = result.theta
        p_j = jnp.asarray(p)
        L = cost_model.lipschitz_constant(theta, p_j)
        if isinstance(L, str):
            raise RuntimeError(f"cost_model.lipschitz_constant: {L}")
        alpha = 0.99 / max(float(L), 1e-12)
        sol = surrogate_eq.solve_extragrad(p_j, theta, x0=x0, alpha=alpha,
                                           tol=1e-10, maxiter=extragrad_max_iter, verbose=0)
        sol.lipschitz_constant = float(L)
        return sol.x, sol
    if USE_POTENTIAL_IN_SURROGATE:
        sol = surrogate_eq.solve(p, result.theta, x0=x0, verbose=0, **SURROGATE_SOLVE_OPTS)
        return sol.x, sol
    if m > 0:
        surrogate_eq.constraints['b'] = b_true0 + S_true @ p
    return surrogate_eq.solve_quadratic(p, result.theta).x, None


N_test_s = len(P_test)
X_sur = np.empty((N_test_s, nvar))
# KKT root-find diagnostics (only populated when USE_POTENTIAL_IN_SURROGATE,
# i.e. surrogate_eq.solve()'s nonlinear root-find is actually used -- the
# quadratic branch solves a QP in closed form and is feasible to machine
# precision by construction). Used below to explain any outliers in
# stats_sur.viol_max.
sur_norm_res = np.full(N_test_s, np.nan)
sur_kkt_evals = np.full(N_test_s, -1, dtype=int)
sur_max_nfev = (extragrad_max_iter if USE_EXTRAGRAD  # extragrad_nlgnep's own maxiter default
                else SURROGATE_SOLVE_OPTS.get('max_nfev', 200))  # nashopt.GNEP.solve() default
# Lipschitz constant L(p) of the learned pseudogradient (Proposition 4.2),
# only computed/used when USE_EXTRAGRAD sets the extragradient step size.
sur_lipschitz = np.full(N_test_s, np.nan)

if COMPARE_GNE_GROUND_TRUTH:
    X_true = np.empty((N_test_s, nvar))
    t0 = time.time()
    for k in tqdm(range(N_test_s), desc="ground truth (solve_gne)", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        X_true[k] = mp.solve_gne(P_test[k], verbose=0).x
    t_true = time.time() - t0

t0 = time.time()
pbar = tqdm(range(N_test_s), desc=surrogate_desc, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}")
for k in pbar:
    if 1:
        # Warm-start the KKT root-find/extragradient iteration from xhat(p)
        # (Part 2c): the SURROGATE game's OWN explicit-NN approximation of
        # its solution p -> x*(p), fit on best-response data generated from
        # the surrogate's own cost model -- a better-matched initial guess
        # than X_mpfit[k] (the TRUE game's explicit solution, used
        # previously) since xhat tracks the surrogate's own equilibrium map
        # instead of the true game's, instead of solve()'s default x0=0: a
        # zero start can leave scipy's 'hybr' reporting success from a
        # stalled, still-infeasible KKT residual at a handful of test points,
        # which would otherwise drag viol_max well above viol_mean for the
        # surrogate.
        x0 = X_hat_sur[k]
    else:
        x0 = np.zeros(nvar)
    
    X_sur[k], sol = solve_surrogate_gne(P_test[k], x0=x0)
    if sol is not None:
        sur_norm_res[k] = sol.norm_residual
        sur_kkt_evals[k] = sol.stats.kkt_evals
    if USE_EXTRAGRAD:
        sur_lipschitz[k] = sol.lipschitz_constant
        pbar.set_postfix(L=f"{sol.lipschitz_constant:.4e}")
t_sur = time.time() - t0
# The extragradient warm start x0 = X_hat_sur[k] above is itself the output
# of xhat's explicit-NN prediction (Part 2c), whose cost (t_predict_sur,
# measured once for the whole test batch) is otherwise not counted anywhere
# in t_sur. Attribute it to the surrogate's reported solve time when
# USE_EXTRAGRAD, so the "per-sample solve time" comparison in the table
# below is a fair total (mpfit's own prediction cost is already counted in
# its own res.t_predict/N_test_s row).
t_sur_reported = t_sur + t_predict_sur if USE_EXTRAGRAD else t_sur

if USE_EXTRAGRAD:
    print(f"\nLipschitz constant L(p) of the learned pseudogradient "
          f"(Proposition 4.2) over the {N_test_s} test points: "
          f"mean = {sur_lipschitz.mean():.4e}, max = {sur_lipschitz.max():.4e} "
          f"(extragradient step size alpha = 0.99/L)")

if COMPARE_GNE_GROUND_TRUTH:
    print(f"\nGround-truth GNE solved via mp.solve_gne (nashopt KKT) for "
          f"{N_test_s} test points in {t_true:.2f} s")
print(f"{surrogate_desc} solved for {N_test_s} test points in {t_sur:.2f} s"
      + (f" ({t_sur_reported:.2f} s incl. the {t_predict_sur:.2f} s spent "
         f"predicting the warm-start xhat)" if USE_EXTRAGRAD else ""))

if USE_POTENTIAL_IN_SURROGATE or USE_EXTRAGRAD:
    # (1) Diagnose the worst surrogate-GNE constraint violations: recompute
    # each X_sur[k]'s own violation of the shared/box constraints and report
    # the KKT root-find diagnostics (residual norm, function evaluations
    # used) at the top offenders, to check whether they stem from a
    # non-/poorly-converged root-find (large norm_residual, and/or
    # kkt_evals hitting the max_nfev cap) rather than surrogate model error.
    viol_lin_k = (np.max(X_sur @ A_true.T - (b_true0[:, None] + S_true @ P_test.T).T, axis=1)
                  if m > 0 else np.full(N_test_s, -np.inf))
    viol_box_k = np.maximum(np.max(lb - X_sur, axis=1), np.max(X_sur - ub, axis=1))
    viol_k = np.maximum(viol_lin_k, viol_box_k)
    worst = np.argsort(viol_k)[::-1][:5]
    print(f"\nWorst {surrogate_desc} constraint violations (top {len(worst)} test points):")
    for k in worst:
        hit_cap = " (hit max_nfev)" if sur_kkt_evals[k] == sur_max_nfev else ""
        evals_label = "extragrad evals" if USE_EXTRAGRAD else "kkt_evals"
        # Under USE_EXTRAGRAD, sur_norm_res[k] is the extragradient method's
        # own stopping-criterion step norm ||x^(K+1)-x^K||_inf at the last
        # iteration K performed (EquilibriumSolver.solve_extragrad's
        # norm_residual), not a KKT residual -- label it accordingly.
        res_label = "||x^(K+1)-x^K||_inf" if USE_EXTRAGRAD else "||KKT residual||"
        print(f"  k={k:4d}  viol={viol_k[k]:.4e}  "
              f"{res_label}={sur_norm_res[k]:.4e}  "
              f"{evals_label}={sur_kkt_evals[k]}{hit_cap}")

# =====================================================================
# Part 4: comparison -- best-response error, constraint violation (all via
# mpfit's own compute_br()/evaluate(), so all three solutions are scored
# identically against the TRUE cost/best-response), and GNE error against
# the ground-truth solution X_true.
# =====================================================================


def evaluate_solution(X, P, name):
    """Best-response error and constraint violation of X(P), via mpfit's own
    compute_br() (uses best_response_fun above) and evaluate()."""
    X_br, Feas = mp.compute_br(X, P, desc=name)
    J = np.array([[float(mp.f[i](x, p)) for i in range(N)] for x, p in zip(X, P)])
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

print("\nEvaluating best-response error and constraint violation on the "
      f"common test set ({N_test_s} samples) for all three solutions...")
stats_mpfit = evaluate_solution(X_mpfit, P_test, "mpfit (explicit NN)")
stats_sur = evaluate_solution(X_sur, P_test, surrogate_desc)

if COMPARE_GNE_GROUND_TRUTH:
    stats_true = evaluate_solution(X_true, P_test, "ground truth (solve_gne)")
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
# F2=surrogate pseudogradient, x2=X_sur. Only meaningful while agent_cost(i)
# is quadratic (a constant-matrix operator to compare against) and while
# COMPARE_GNE_GROUND_TRUTH computed gne_err_sur above. F2 is computed via
# cost_model.pseudogradient(x,p,theta) (generic autodiff over the full
# theta), not just A_hat(p) @ x + q_hat(p): with USE_POTENTIAL_IN_SURROGATE
# True this also captures grad_x Psi(x,p;theta_pot) (0 contribution when
# the flag is False). Aq='NN' makes A_hat(p), q_hat(p) p-dependent, so F2 is
# recomputed at each test p (the p=0 snapshot from Part 2a is not reused,
# unlike under Aq='constant' where it would be exact for every p).
# =====================================================================

if COST_TYPE == 'quadratic' and COMPARE_GNE_GROUND_TRUTH:
    M_true = np.zeros((nvar, nvar))
    for i in range(N):
        M_true[i1_p[i]:i2_p[i], :] = Q_true[i][i1_p[i]:i2_p[i], :]
    rel_err_A = np.linalg.norm(A_hat0 - M_true) / np.linalg.norm(M_true)

    mismatch_norms = np.empty(N_test_s)
    for k in range(N_test_s):
        p = P_test[k]
        f_true_p = np.concatenate([(c_true0[i] + F[i] @ p)[i1_p[i]:i2_p[i]] for i in range(N)])
        F_true_at_xsur = M_true @ X_sur[k] + f_true_p
        F_sur_at_xsur = np.asarray(
            cost_model.pseudogradient(jnp.asarray(X_sur[k]), jnp.asarray(p), result.theta))
        mismatch_norms[k] = np.linalg.norm(F_true_at_xsur - F_sur_at_xsur)

    bound_rhs = mismatch_norms / mu  # mu = 0.1, the TRUE game's certified strong-monotonicity constant

    print("\n" + "=" * 78)
    print("Diagnostic: pseudogradient (operator) mismatch vs. cost-value R2")
    print("=" * 78)
    print(f"Quadratic part relative error at p=0  ||A_hat(0) - Q_true|| / ||Q_true|| "
          f"(raw, unsymmetrized) = {rel_err_A:.4e}")
    print(f"Operator mismatch ||F_true(X_sur)-F_sur(X_sur)|| (mean/max)  = "
          f"{mismatch_norms.mean():.4e} / {mismatch_norms.max():.4e}")
    print(f"VI sensitivity bound  ||mismatch||/mu             (mean/max) = "
          f"{bound_rhs.mean():.4e} / {bound_rhs.max():.4e}")
    print(f"Observed GNE error vs ground truth                (mean/max) = "
          f"{gne_err_sur.mean():.4e} / {gne_err_sur.max():.4e}")
    print("=" * 78)
else:
    reasons = []
    if COST_TYPE != 'quadratic':
        reasons.append(f"COST_TYPE = {COST_TYPE!r} (no constant-matrix true operator)")
    if not COMPARE_GNE_GROUND_TRUTH:
        reasons.append("COMPARE_GNE_GROUND_TRUTH = False (no ground-truth GNE to compare against)")
    print(f"\nSkipping pseudogradient-mismatch diagnostic ({'; '.join(reasons)}).")

print("\n" + "=" * 78)
print("Comparison on the common test set "
      f"({N_test_s} samples): mpfit (explicit NN) vs. StructuredMonotoneCost "
      f"{surrogate_desc}")
print("=" * 78)
header = f"{'':32s}{'mpfit (NN)':>18s}{'surrogate':>22s}"
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
      f"{res.t_predict / N_test_s:18.2e}{t_sur_reported / N_test_s:22.2e}")
print("=" * 78)
if COMPARE_GNE_GROUND_TRUTH:
    print("Ground-truth solve sanity check (mp.solve_gne on the true game): "
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
    """Format a float in scientific notation as LaTeX, matching the terminal's
    {:.{decimals}e} style, e.g. 0.54474 -> $5.4474 \\cdot 10^{-1}$ (or
    $\\mathbf{5.4474 \\cdot 10^{-1}}$ if bold=True)."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    body = f"{mantissa} \\cdot 10^{{{int(exponent)}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"$ {body} $"


rows = [
    # (name, mpfit value, surrogate value, decimals, higher_is_better, fmt)
    # exponential notation (matching the terminal's {:.4e} printout), since
    # these quantities are near zero and fixed-point notation at 4 decimals
    # would otherwise round them all down to 0.0000
    ("best-response error (mean)", stats_mpfit.br_err_mean_feasible, stats_sur.br_err_mean_feasible, 4, False, "exp"),
    ("constraint violation (mean)", stats_mpfit.viol_mean, stats_sur.viol_mean, 4, False, "exp"),
    ("constraint violation (max)", stats_mpfit.viol_max, stats_sur.viol_max, 4, False, "exp"),
]
if COMPARE_GNE_GROUND_TRUTH:
    rows += [
        ("GNE error vs ground truth (mean)", gne_err_mpfit.mean(), gne_err_sur.mean(), 4, False, "exp"),
        ("GNE error vs ground truth (max)", gne_err_mpfit.max(), gne_err_sur.max(), 4, False, "exp"),
    ]
rows += [
    ("opt.cost/cost fit (average R$^2$ score)", r2_mpfit, r2_sur, 4, True, "fixed"),
    ("fitting time (s)", res.training_time, surrogate_fit_time, 2, False, "fixed"),
    ("per-sample solve time (s)", res.t_predict / N_test_s, t_sur_reported / N_test_s, 2, False, "exp"),
]

latex_lines = [
    "% Auto-generated by example_surrogate_vs_explicit_NL.py",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Approximate explicit GNE solution~\\cite{BT26} vs. monotone-by-construction" +
    (" surrogate (with potential term)" if USE_POTENTIAL_IN_SURROGATE else " quadratic surrogate") +
    " on a nonlinear-cost game.}",
    "\\label{tab:surrogate_vs_explicit_nl}",
    "\\setlength{\\tabcolsep}{5pt}",
    "\\renewcommand{\\arraystretch}{1.} % Adjust row separation",
    "\\begin{tabular}{l|r|r}",
    "\\toprule",
    " & explicit~\\cite{BT26} & surrogate GNE \\\\",
    "\\midrule",
]
for name, v_mpfit, v_sur, decimals, higher_is_better, fmt in rows:
    bold_mpfit = v_mpfit >= v_sur if higher_is_better else v_mpfit <= v_sur
    bold_sur = v_sur >= v_mpfit if higher_is_better else v_sur <= v_mpfit
    fmt_fun = exp_to_latex if fmt == "exp" else sci_to_latex
    latex_lines.append(
        f"{name} & {fmt_fun(v_mpfit, decimals, bold_mpfit)} & "
        f"{fmt_fun(v_sur, decimals, bold_sur)} \\\\")
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
