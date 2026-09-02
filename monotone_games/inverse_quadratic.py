"""Closed-form fast paths for inverse-learning a quadratic monotone game from
best-response data [1, Section 2.2], bypassing GameLearner's generic
bilevel/autodiff route.

[1] A. Bemporad and T. Tatarenko, "Learning Parametric Monotone Games,"
    arXiv preprint, 2026.

(C) 2026 A. Bemporad
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.models import StaticModel
from joblib import cpu_count

from .cost_models import StructuredMonotoneCost


class QuadraticInverseSolver:
    """Fits a quadratic monotone game directly from best-response samples,
    exploiting the affine best-response structure of quadratic monotone
    games (Sec. 5): for costs J_i(x) = .5 x_i^T A_ii x_i + x_i^T A_{i,-i} x_{-i}
    + q_i^T x_i (+ terms independent of x_i), the unconstrained best response is

        bar_x_i = -A_ii^{-1}(A_{i,-i} x_{-i} + q_i),   q(p) = q0 + q1@p affine in p.

    fit_nls parameterizes A(p) = C(p)^T C(p) + mu I + D(p) - D(p)^T, monotone
    by construction for every p, with the nonzero entries of C, D either
    affine in p (Aq='affine', matching StructuredMonotoneCost's 'affine') or
    constant in p (Aq='constant': C, D fixed, only q(p) = q1 p + q0 varies).
    The SDP paths fit_sdp/fit_ls_sdp_cascade always take A constant in p, as
    required for convexity of those formulations (eq. sdp-monotone /
    two-stage), independently of this Aq setting, which only affects
    fit_nls/nls_initial_guess/to_cost_model.

    Xbr, P, agent_idx have the same layout as BestResponseLoss's data: Xbr[k,:]
    is the observed decision vector of all agents at sample k, P[k,:] the
    parameter value, and agent_idx[k] identifies which agent's slice is the
    observed best response (the other slices provide x_{k,-i}).
    """

    def __init__(self, dims, npar, mu=0., Aq='affine'):
        if Aq not in ('affine', 'constant'):
            raise ValueError(f"Unknown Aq type {Aq!r}: must be 'affine' or 'constant'")
        self.dims = list(dims)
        self.N = len(dims)
        self.nx = int(sum(dims))
        self.npar = npar
        self.mu = mu
        self.Aq = Aq
        model = StructuredMonotoneCost(dims, npar, mu=mu, potential=None)
        self.isi = model.isi
        self.nisi = model.nisi
        self.mask_C = model.mask_C
        self.mask_D = model.block_mask_D
        self._nC = int(self.mask_C.sum())
        self._nD = int(self.mask_D.sum())
        self._idx_C = np.where(self.mask_C)
        self._idx_D = np.where(self.mask_D)

    # -- theta = [C1,C2,D1,D2,q0,q1] (Aq='affine') or [C2,D2,q0,q1]
    # (Aq='constant'); used by fit_nls only -----------------------------------
    # theta collects only the free entries: the upper-triangular part of
    # C(p), the block strictly upper-triangular part of D(p), and q0, q1
    # (eq. pseudogradient-matrix).
    def _theta_shapes(self):
        nx, npar, nC, nD = self.nx, self.npar, self._nC, self._nD
        if self.Aq == 'affine':
            return [(nC, npar), (nC,), (nD, npar), (nD,), (nx,), (nx, npar)]
        else:  # 'constant'
            return [(nC,), (nD,), (nx,), (nx, npar)]

    def _A_of_p(self, C2, D2, p, C1=None, D1=None):
        """A(p) = C(p)^T C(p) + mu I + D(p) - D(p)^T (numpy). C1/D1=None means
        Aq='constant': C(p)=C2, D(p)=D2, independent of p."""
        C = np.zeros((self.nx, self.nx))
        C[self.mask_C] = C2 if C1 is None else C1 @ p + C2
        D = np.zeros((self.nx, self.nx))
        D[self.mask_D] = D2 if D1 is None else D1 @ p + D2
        return C.T @ C + self.mu * np.eye(self.nx) + D - D.T

    def _build_br_loss(self, Xbr, P, agent_idx):
        """Build a jitted function loss(theta) -> mean best-response residual
        (1/(2K)) sum_k ||bar_x_{i_k}(theta;p_k,x_{k,-i_k}) - x_{k,i_k}||^2 over
        the K samples in (Xbr,P,agent_idx), theta as in _theta_shapes. The 1/2
        factor matches the 0.5*sum(residuals**2) convention of nonlinear-
        least-squares solvers (e.g. scipy.optimize.least_squares), so
        rho_th=rho/2 reproduces the paper's rho/2 ||theta||^2 term exactly.
        """
        Xbr, P, agent_idx = np.asarray(Xbr), np.asarray(P), np.asarray(agent_idx)
        isi, nisi, idx_C, idx_D, mu, nx = (self.isi, self.nisi, self._idx_C,
                                          self._idx_D, self.mu, self.nx)
        K = Xbr.shape[0]
        rows_by_agent = [np.where(agent_idx == i)[0] for i in range(self.N)]
        affine = self.Aq == 'affine'

        if affine:
            def A_of_p(C1, C2, D1, D2, p):
                C = jnp.zeros((nx, nx)).at[idx_C].set(C1 @ p + C2)
                D = jnp.zeros((nx, nx)).at[idx_D].set(D1 @ p + D2)
                return C.T @ C + mu * jnp.eye(nx) + D - D.T
        else:  # 'constant': C, D independent of p
            def A_of_p(C2, D2, p):
                C = jnp.zeros((nx, nx)).at[idx_C].set(C2)
                D = jnp.zeros((nx, nx)).at[idx_D].set(D2)
                return C.T @ C + mu * jnp.eye(nx) + D - D.T

        @jax.jit
        def loss(theta):
            if affine:
                C1, C2, D1, D2, q0, q1 = theta
                Am = jax.vmap(A_of_p, in_axes=(None, None, None, None, 0))(C1, C2, D1, D2, P)
            else:
                C2, D2, q0, q1 = theta
                Am = jax.vmap(A_of_p, in_axes=(None, None, 0))(C2, D2, P)
            Qm = P @ q1.T + q0
            total = 0.
            for i, rows in enumerate(rows_by_agent):
                if rows.size == 0:
                    continue
                Aii = Am[np.ix_(rows, isi[i], isi[i])]
                Ani = Am[np.ix_(rows, isi[i], nisi[i])]
                rhs = jnp.einsum('kij,kj->ki', Ani, Xbr[np.ix_(rows, nisi[i])]) + Qm[np.ix_(rows, isi[i])]
                br = -jnp.linalg.solve(Aii, rhs[..., None])[..., 0]
                total = total + jnp.sum((br - Xbr[np.ix_(rows, isi[i])]) ** 2)
            return total / (2. * K)
        return loss

    def fit_nls(self, Xbr, P, agent_idx, seeds=0, rho=1.e-8, warm_start=None,
               adam_epochs=1000, lbfgs_epochs=5000, val_data=None, n_jobs=None):
        """Nonlinear least-squares fit (eq. nls-monotone) via jax_sysid's
        StaticModel (Adam + L-BFGS-B), directly on the affine best-response
        residual bar_x_i - x_{k,i}, with A(p) parameterized as in the class
        docstring (Aq='affine' or 'constant', monotone by construction
        either way) and the paper's rho/2 ||theta||^2 regularization.

        seeds : int or array-like of ints
            Random seed(s) for the initial guess. A single seed (default)
            runs one fit from a random initialization. Multiple seeds run
            one fit_nls per seed in parallel (jax_sysid's StaticModel.
            parallel_fit / joblib), keeping the fit with the lowest score
            (see val_data). Each run is initialized from its own seed, except
            that with more than one seed and warm_start given, the first
            seed's run starts from warm_start instead (per the paper's note
            that the two-stage solution can initialize the NLS problem); a
            single-seed run never uses warm_start.
        warm_start : SimpleNamespace or None
            A fit_sdp/fit_ls_sdp_cascade result, converted via
            nls_initial_guess; see seeds above for when it is actually used.
        val_data : (Xbr_val, P_val, agent_idx_val) or None
            Held-out best-response data used to select the best of len(seeds)
            > 1 parallel fits by held-out residual; if None, the fit with the
            lowest training loss is kept instead.
        n_jobs : int or None
            Number of parallel jobs when len(seeds) > 1 (default: cpu_count()).

        Returns a SimpleNamespace with the fitted factors (C1,C2,D1,D2,q0,q1
        if Aq='affine', else C2,D2,q0,q1), the fitted jax_sysid model, and
        the callables A_of_p(p), q_of_p(p).
        """
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)

        if isinstance(seeds, (int, np.integer)):
            seeds = [int(seeds)]
        else:
            seeds = [int(s) for s in seeds]

        shapes = self._theta_shapes()

        def random_init(seed):
            rng = np.random.RandomState(seed)
            return [rng.standard_normal(shape) for shape in shapes]

        warm_theta = (self.nls_initial_guess(warm_start)
                     if warm_start is not None and len(seeds) > 1 else None)

        def init_fcn(seed):
            if warm_theta is not None and seed == seeds[0]:
                return warm_theta
            return random_init(seed)

        data_loss = self._build_br_loss(Xbr, P, agent_idx)

        def output_fcn(U, theta):
            return jnp.zeros((U.shape[0], 1))

        model = StaticModel(1, self.nx + self.npar, output_fcn)
        model.loss(rho_th=rho / 2., custom_regularization=data_loss)
        model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        M = np.asarray(Xbr).shape[0]
        Y_dummy, U_dummy = np.zeros((M, 1)), np.zeros((M, self.nx + self.npar))

        if len(seeds) > 1:
            models = model.parallel_fit(Y_dummy, U_dummy, init_fcn, seeds=seeds,
                                        n_jobs=n_jobs or cpu_count())
            if val_data is not None:
                score_fn = self._build_br_loss(*val_data)
                scores = [float(score_fn(m.params)) for m in models]
            else:
                scores = [float(data_loss(m.params)) for m in models]
            best = models[int(np.argmin(scores))]
        else:
            model.init(params=init_fcn(seeds[0]))
            model.fit(Y_dummy, U_dummy)
            best = model

        params = [np.asarray(th) for th in best.params]
        if self.Aq == 'affine':
            C1, C2, D1, D2, q0, q1 = params
            return SimpleNamespace(
                C1=C1, C2=C2, D1=D1, D2=D2, q0=q0, q1=q1, model=best,
                A_of_p=lambda p: self._A_of_p(C2, D2, np.asarray(p), C1=C1, D1=D1),
                q_of_p=lambda p: q0 + q1 @ np.asarray(p))
        else:  # 'constant'
            C2, D2, q0, q1 = params
            return SimpleNamespace(
                C2=C2, D2=D2, q0=q0, q1=q1, model=best,
                A_of_p=lambda p: self._A_of_p(C2, D2, np.asarray(p)),
                q_of_p=lambda p: q0 + q1 @ np.asarray(p))

    def nls_initial_guess(self, fit_result):
        """Build a fit_nls initial theta from a constant-A fit result
        (fit_sdp/fit_ls_sdp_cascade), per the paper's note that the two-stage
        solution can initialize the NLS problem: C, D are recovered from A
        as in to_cost_model. If Aq='affine', the p-dependent parts C1, D1
        are set to 0; if Aq='constant', theta has no C1/D1 slots to begin
        with.
        """
        A = np.asarray(fit_result.A)
        sym = 0.5 * (A + A.T) - self.mu * np.eye(self.nx)
        sym = 0.5 * (sym + sym.T)
        eigval, eigvec = np.linalg.eigh(sym)
        eigval = np.clip(eigval, 0., None)
        _, C = np.linalg.qr(np.diag(np.sqrt(eigval)) @ eigvec.T)
        D = np.where(self.mask_D, 0.5 * (A - A.T), 0.)
        if self.Aq == 'affine':
            return [np.zeros((self._nC, self.npar)), C[self.mask_C],
                    np.zeros((self._nD, self.npar)), D[self.mask_D],
                    np.asarray(fit_result.q0).ravel(), np.asarray(fit_result.q1)]
        else:  # 'constant'
            return [C[self.mask_C], D[self.mask_D],
                    np.asarray(fit_result.q0).ravel(), np.asarray(fit_result.q1)]

    def fit_sdp(self, Xbr, P, agent_idx, rho=0., **solver_opts):
        """Convex SDP fit (eq. sdp-monotone) via cvxpy: weights each
        best-response residual by A_ii and imposes mu-monotonicity, symmetric
        diagonal blocks, and a trace normalization directly as convex
        constraints, solved to global optimality. rho adds the paper's
        rho/2 ||theta||^2 regularization (default 0: the tr(A)=n constraint
        already rules out the trivial solution). Requires cvxpy.
        """
        import cvxpy as cp
        Xbr, P, agent_idx = np.asarray(Xbr), np.asarray(P), np.asarray(agent_idx)
        nx, npar, mu = self.nx, self.npar, self.mu
        isi, nisi = self.isi, self.nisi

        A = cp.Variable((nx, nx))
        q0 = cp.Variable(nx)
        q1 = cp.Variable((nx, npar))

        residual_terms = []
        for x, p, i in zip(Xbr, P, agent_idx):
            Aii = A[np.ix_(isi[i], isi[i])]
            Ani = A[np.ix_(isi[i], nisi[i])]
            qi0 = q0[isi[i]]
            qi1 = q1[isi[i], :]
            residual_terms.append(cp.sum_squares(Ani @ x[nisi[i]] + qi0 + qi1 @ p + Aii @ x[isi[i]]))
        objective = cp.sum(residual_terms) / len(Xbr)
        if rho > 0.:
            objective = objective + rho / 2. * (cp.sum_squares(A) + cp.sum_squares(q0) + cp.sum_squares(q1))

        constraints = [0.5 * (A + A.T) >> mu * np.eye(nx), cp.trace(A) == nx]
        for i in range(self.N):
            Aii = A[np.ix_(isi[i], isi[i])]
            constraints.append(Aii == Aii.T)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(**solver_opts)
        return SimpleNamespace(A=A.value, q0=q0.value, q1=q1.value, problem=problem)

    def fit_ls_sdp_cascade(self, Xbr, P, agent_idx, **solver_opts):
        """Two-stage fit (eq. ls / two-stage): an unconstrained per-agent
        ordinary-least-squares fit of (P_i,f_i0,f_i1) from eq. ls (decouples
        across agents), followed by a small cvxpy SDP (eq. two-stage)
        reconciling A,q0,q1 with the OLS estimates -- of size independent of
        the number of samples K. Requires cvxpy.
        """
        import cvxpy as cp
        Xbr, P, agent_idx = np.asarray(Xbr), np.asarray(P), np.asarray(agent_idx)
        nx, npar = self.nx, self.npar
        isi, nisi = self.isi, self.nisi

        Ps, f0s, f1s = [], [], []
        for i in range(self.N):
            rows = np.where(agent_idx == i)[0]
            nmi = len(nisi[i])
            Z = np.hstack([Xbr[rows][:, nisi[i]], np.ones((len(rows), 1)), P[rows]])
            Y = Xbr[rows][:, isi[i]]
            theta_i, *_ = np.linalg.lstsq(Z, Y, rcond=None)
            Ps.append(theta_i[:nmi].T)
            f0s.append(theta_i[nmi])
            f1s.append(theta_i[nmi + 1:].T)

        A = cp.Variable((nx, nx))
        q0 = cp.Variable(nx)
        q1 = cp.Variable((nx, npar))
        residual_terms = []
        for i in range(self.N):
            Aii = A[np.ix_(isi[i], isi[i])]
            Ani = A[np.ix_(isi[i], nisi[i])]
            lhs = Aii @ np.hstack([Ps[i], f0s[i].reshape(-1, 1), f1s[i]])
            rhs = cp.hstack([Ani, cp.reshape(q0[isi[i]], (len(isi[i]), 1), order='F'), q1[isi[i], :]])
            residual_terms.append(cp.sum_squares(lhs + rhs))
        objective = cp.sum(residual_terms)

        constraints = [0.5 * (A + A.T) >> self.mu * np.eye(nx), cp.trace(A) == nx]
        for i in range(self.N):
            Aii = A[np.ix_(isi[i], isi[i])]
            constraints.append(Aii == Aii.T)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(**solver_opts)
        return SimpleNamespace(A=A.value, q0=q0.value, q1=q1.value, problem=problem)

    def to_cost_model(self, fit_result):
        """Package a fit result into a StructuredMonotoneCost(potential=None,
        Aq=self.Aq) instance plus its theta, ready for GameLearner/
        EquilibriumSolver use.

        fit_nls results carry the fitted factors (C1,C2,D1,D2 if Aq='affine',
        else C2,D2) directly. For the constant-A SDP results (fit_sdp/
        fit_ls_sdp_cascade, always constant in p regardless of self.Aq), C
        (upper-triangular) is recovered via a QR factorization of the
        symmetric PSD part of A (Lemma 2's construction run in reverse: a
        square-root factor QR-factorizes into an upper-triangular C with
        C^T C equal to that PSD part), and D (block strictly upper-triangular)
        as the corresponding block of A's skew-symmetric part; if Aq='affine'
        the p-dependent parts C1, D1 are left at 0.

        All theta entries beyond (C,D,q) -- the phi_i nets -- are zero, so
        phi_i is identically 0 (best-response data carries no information
        about the x_i-independent terms anyway).
        """
        model = StructuredMonotoneCost(self.dims, self.npar, mu=self.mu,
                                       potential=None, Aq=self.Aq)
        theta = [np.zeros(shape) for shape in model.param_shapes()]
        affine = self.Aq == 'affine'

        if hasattr(fit_result, "model"):  # fit_nls: theta already shaped per self.Aq
            if affine:  # C(p) = C1 p + C2, D(p) = D1 p + D2
                theta[0][self.mask_C] = fit_result.C1
                theta[1][self.mask_C] = fit_result.C2
                theta[2][self.mask_D] = fit_result.D1
                theta[3][self.mask_D] = fit_result.D2
            else:  # C(p) = C2, D(p) = D2, constant
                theta[0][self.mask_C] = fit_result.C2
                theta[1][self.mask_D] = fit_result.D2
        else:  # constant A from fit_sdp / fit_ls_sdp_cascade
            A = fit_result.A
            sym = 0.5 * (A + A.T) - self.mu * np.eye(self.nx)
            sym = 0.5 * (sym + sym.T)  # symmetrize away solver/roundoff tolerance
            eigval, eigvec = np.linalg.eigh(sym)
            eigval = np.clip(eigval, 0., None)
            B = np.diag(np.sqrt(eigval)) @ eigvec.T
            _, C = np.linalg.qr(B)  # C^T C == B^T B == sym, C upper-triangular
            Cmasked = np.where(self.mask_C, C, 0.)
            Dmasked = np.where(self.mask_D, 0.5 * (A - A.T), 0.)
            if affine:
                theta[1] = Cmasked
                theta[3] = Dmasked
            else:
                theta[0] = Cmasked
                theta[1] = Dmasked

        # q1, q2 (this class's q1, q0) are the last two entries of the Aq
        # parameter block, not of theta as a whole: model uses the default
        # phi='quadratic', which appends 5 more entries after the Aq block,
        # so theta[-2]/theta[-1] would silently overwrite those instead.
        n_Aq = 6 if affine else 4
        theta[n_Aq - 2] = np.asarray(fit_result.q1)
        theta[n_Aq - 1] = np.asarray(fit_result.q0)
        return model, theta
