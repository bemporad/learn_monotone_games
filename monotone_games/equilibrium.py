"""Equilibrium computation for a fitted CostModel, wrapping nashopt.GNEP /
nashopt.lq.qp_gnep (requires nashopt>=1.3.0).

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 
    
(C) 2026 A. Bemporad
"""

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, root


class _ExtragradConstraints:
    """Minimal facade exposing exactly the attributes/methods
    nashopt.nonlinear.nl_extragrad._Projector/_ProjectorTRF need from a
    nashopt.GNEP (nvar, lb, ub, ng, nh, neq, Aeq, g, dg) -- built ONCE by
    EquilibriumSolver.compile_parametric_extragrad() around the JIT-compiled,
    p-TRACED g_fn(x,p)/dg_fn(x,p), with only the box bounds fixed and .p
    (the current test parameter) mutated by solve_extragrad() before each
    project() call. Since g_fn/dg_fn were compiled once as functions of both
    x and p, evaluating self.g(x)/self.dg(x) for a NEW self.p reuses the same
    compiled executable -- no retracing -- unlike rebuilding a fresh
    nashopt.GNEP (and hence fresh jax.jit wrappers) per call.
    """

    def __init__(self, nvar, lb, ub, ng, g_fn, dg_fn):
        self.nvar = nvar
        self.lb = lb
        self.ub = ub
        self.ng = ng
        self.nh = 0
        self.neq = 0
        self.Aeq = None
        self.beq = None
        self._g_fn = g_fn
        self._dg_fn = dg_fn
        self.p = None  # set by solve_extragrad() before every project() call

    def g(self, x):
        return self._g_fn(x, self.p)

    def dg(self, x):
        return self._dg_fn(x, self.p)


class EquilibriumSolver:
    """monotone_games' parametric-GNEP wrapper around a fitted CostModel:
    construct ONCE per (cost_model, constraints) -- e.g. right after
    learner.fit(), or via GameLearner.fit()'s result.equilibrium_solver(...)
    convenience -- then call solve(p, theta, ...)/best_response(p, theta, i,
    x, ...) as many times as needed, for as many different p's as needed,
    at only a root-find's/L-BFGS-B's cost each: the first call for a given
    theta (or an explicit compile_parametric(theta)/compile_parametric_br
    (theta)) JIT-compiles the underlying JAX computation ONCE, treating p as
    a genuine traced argument rather than a value baked into a Python
    closure, and every subsequent call with that SAME theta object reuses
    the compiled executable instead of retracing/recompiling from scratch.
    This is automatic: no separate method needs to be learned or called by
    an example to get the fast path -- solve()/best_response() themselves
    detect a new/changed theta (by identity) and (re)compile transparently,
    so ANY code -- existing or new -- that just calls these two methods in
    a loop over p benefits, with the exact same call convention as a
    one-off solve. Only when theta changes (e.g. retraining) is a
    recompilation triggered again, automatically.

    constraints: optional dict describing shared constraints on x (Sec. 5's
    "generalized monotone games" extension), with keys among 'g','ng' (shared
    nonlinear inequalities g(x,p)<=0), 'A','b' (shared linear A x <= b,
    optionally parametrized by 'S' as A x <= b + S@p), 'lb','ub' (box
    bounds). solve_quadratic (the closed-form path for a pure quadratic,
    potential=None cost model) reads 'A'/'b' as a FIXED (non-parametric)
    constraint, refreshed by the caller per p if needed; solve/best_response
    build the p-dependent constraint g(x,p) = A@x - (b + S@p) once from
    'A'/'b'/'S' (or use 'g'/'ng' directly, if 'g' already has the (x,p)
    signature) so its p-dependence is captured by the JIT trace instead of
    requiring the constraint to be refreshed on every call.
    """

    def __init__(self, cost_model, constraints=None):
        self.cost_model = cost_model
        self.constraints = constraints or {}
        self._parametric = None  # set by compile_parametric(); .theta identifies the compiled theta
        self._br = None  # set by compile_parametric_br(); .theta identifies the compiled theta
        self._extragrad = None  # set by compile_parametric_extragrad(); .theta identifies the compiled theta

    def _nonsmooth_potential(self):
        spec = getattr(self.cost_model, "potential_spec", None)
        return spec is not None and spec["type"] in ("PWA", "PWQ")

    def _build_shared_g(self, npar):
        """Return (g, ng) for the shared inequality constraint g(x,p)<=0:
        self.constraints['g']/['ng'] if given, else built from 'A'/'b'
        (optional 'S') as A@x <= b + S@p, else (None, 0). Shared by
        compile_parametric and compile_parametric_br so both build exactly
        the same constraint from the same constraints dict.
        """
        g, ng = self.constraints.get('g'), self.constraints.get('ng')
        A = self.constraints.get('A')
        if g is None and A is not None:
            A_j = jnp.asarray(A)
            b_j = jnp.asarray(self.constraints['b'])
            S = self.constraints.get('S')
            S_j = jnp.asarray(S) if S is not None else jnp.zeros((A_j.shape[0], npar))
            g = lambda x, p: A_j @ x - (b_j + S_j @ p)
            ng = A_j.shape[0]
        return g, int(ng) if ng is not None else 0

    # ------------------------------------------------------------------
    # Joint GNE solve: KKT stationarity + Fischer-Burmeister complementarity,
    # root-found via scipy.optimize.root.
    # ------------------------------------------------------------------

    def compile_parametric(self, theta):
        """JIT-compile ONCE the KKT stationarity + Fischer-Burmeister
        complementarity residual (and its Jacobian) of a nashopt.GNEP built
        from cost_model.costs(.,.,theta), with p a genuine JAX-traced
        argument rather than a value baked into a Python closure.
        Idempotent: a repeat call with the SAME theta object is a no-op;
        solve() calls this automatically on first use (or whenever theta
        changes), so calling it explicitly is only needed to report
        compile time separately from the first solve() call. Returns self,
        so it can be chained, e.g.
        `eq_solver = EquilibriumSolver(cost_model, constraints).compile_parametric(theta)`.
        """
        if self._parametric is not None and self._parametric.theta is theta:
            return self  # already compiled for this exact theta object

        from nashopt import GNEP

        cost_model = self.cost_model
        N = cost_model.N
        npar = cost_model.npar

        if self._nonsmooth_potential():
            raise NotImplementedError(
                "Automatic epigraph/slack reformulation for nonsmooth (PWA/PWQ) "
                "potentials (Sec. 4.1) is not implemented; solve manually by "
                "adding N slack variables y_i and shared constraints "
                "psi_j(x) <= y_h to nashopt.GNEP's g/ng shared-constraint "
                "interface.")

        f = [lambda x, p, i=i: cost_model.costs(x, p, theta)[i] for i in range(N)]
        g, ng = self._build_shared_g(npar)

        gnep = GNEP(sizes=cost_model.dims, f=f, g=g, ng=ng,
                    lb=self.constraints.get('lb'), ub=self.constraints.get('ub'),
                    parametric=True)
        # GNEP.__init__ always sets self.npar = 0, regardless of
        # parametric=True (only its ParametricGNEP subclass -- a different,
        # design-oriented solve() for optimizing over p -- sets it correctly
        # in the same way). Setting it here activates the p-aware KKT
        # residual (kkt_residual_shared/kkt_residual_i) that GNEP already
        # implements correctly once self.npar is right, without depending on
        # ParametricGNEP's solve().
        gnep.npar = npar

        def kkt_residual_p(z, p):
            # z = [x, lam] (the unknowns solved for); p enters as a separate
            # traced argument held fixed by root(), not as part of the
            # unknown -- unlike nashopt's own p-in-z convention (meant for
            # ParametricGNEP.solve()'s design problem), so the KKT system
            # solved here stays exactly the same square system a one-off
            # solve at a fixed p would solve.
            return gnep.kkt_residual(jnp.concatenate((z, p)))

        kkt_fun = jax.jit(kkt_residual_p)
        kkt_jac = jax.jit(jax.jacobian(kkt_residual_p, argnums=0))

        # Trigger the one-time trace/compile now, so its cost is attributed
        # to compile_parametric() rather than inflating the first solve()
        # call.
        z0 = jnp.zeros(gnep.nvar + gnep.nlam_sum)
        p0 = jnp.zeros(npar)
        kkt_fun(z0, p0).block_until_ready()
        kkt_jac(z0, p0).block_until_ready()

        self._parametric = SimpleNamespace(theta=theta, gnep=gnep, kkt_fun=kkt_fun, kkt_jac=kkt_jac)
        return self

    def solve(self, p, theta, x0=None, solver='hybr', tol=1e-12, max_nfev=200, verbose=0):
        """Solve for x*(p): the joint GNE of cost_model.costs(.,.,theta)
        under the constraints set at construction. Compiles the underlying
        KKT residual/Jacobian ONCE per (instance, theta) -- via
        compile_parametric(theta), called automatically here if not already
        done -- and reuses that compiled executable for every subsequent
        call with the SAME theta object, regardless of p, solver, tol,
        max_nfev, or verbose (none of which affect the compiled trace).
        Returns a SimpleNamespace (fields .x, .lam, .res, .stats with
        .kkt_evals, .norm_residual), matching nashopt.GNEP.solve()'s shape.
        """
        self.compile_parametric(theta)
        from nashopt._common.report import eval_residual

        gnep = self._parametric.gnep
        kkt_fun, kkt_jac = self._parametric.kkt_fun, self._parametric.kkt_jac
        p = jnp.asarray(p)
        x0 = jnp.zeros(gnep.nvar) if x0 is None else jnp.asarray(x0)
        lam0 = 0.1 * jnp.ones(gnep.nlam_sum) if gnep.has_constraints else jnp.zeros(0)
        z0 = jnp.concatenate((x0, lam0))

        def fun(z):
            return kkt_fun(z, p)

        def jac(z):
            return kkt_jac(z, p)

        options = dict(xtol=tol, maxfev=max_nfev)
        try:
            solution = root(fun, z0, jac=jac, method=solver, options=options)
        except Exception as e:
            raise RuntimeError(
                f"Error in root solver (method='{solver}'): {str(e)} "
                "The KKT system may be non-square for this problem "
                "configuration (non-variational GNE with shared equality "
                "constraints and more than one agent; try solver='lm' or "
                "'trf'/'dogbox').") from e

        z_star = np.asarray(solution.x)
        res = np.asarray(fun(z_star))
        kkt_evals = solution.nfev
        x = z_star[:gnep.nvar]
        lam = []
        if gnep.has_constraints:
            lam_star = z_star[gnep.nvar:]
            for i in range(gnep.N):
                lam.append(lam_star[gnep.ii_lam[i]])

        norm_res = eval_residual(res, verbose, kkt_evals, 0.)
        stats = SimpleNamespace(solver=solver, kkt_evals=kkt_evals, elapsed_time=0.)
        return SimpleNamespace(x=x, res=res, lam=lam, stats=stats, norm_residual=norm_res)

    # ------------------------------------------------------------------
    # Per-agent best response: min_{x_i} f_i(x,p;theta) + penalty(g(x,p)),
    # box-constrained, solved via scipy.optimize.minimize(L-BFGS-B).
    # ------------------------------------------------------------------

    def compile_parametric_br(self, theta):
        """JIT-compile ONCE, for every agent i, the value-and-gradient of
        the penalized best-response objective

            obj_i(xi, x, p, rho) = f_i(x with x_i replaced by xi, p; theta)
                                    + rho * sum(max(g(x_i_replaced, p), 0)**2)

        with (xi, x, p, rho) all genuine JAX-traced arguments (only the
        agent index i is static/Python-level, since agents may have
        different dims). Idempotent: a repeat call with the SAME theta
        object is a no-op; best_response() calls this automatically on
        first use (or whenever theta changes), so calling it explicitly is
        only needed to report compile time separately from the first
        best_response() call. Returns self, so it can be chained, e.g.
        `eq_solver.compile_parametric_br(theta)`.
        """
        if self._br is not None and self._br.theta is theta:
            return self  # already compiled for this exact theta object

        if self._nonsmooth_potential():
            raise NotImplementedError(
                "Automatic epigraph/slack reformulation for nonsmooth (PWA/PWQ) "
                "potentials (Sec. 4.1) is not implemented; solve manually by "
                "adding N slack variables y_i and shared constraints "
                "psi_j(x) <= y_h to nashopt.GNEP's g/ng shared-constraint "
                "interface.")

        cost_model = self.cost_model
        N = cost_model.N
        npar = cost_model.npar
        g, ng = self._build_shared_g(npar)

        lb = self.constraints.get('lb')
        ub = self.constraints.get('ub')
        lb = np.asarray(lb) if lb is not None else -np.inf * np.ones(cost_model.nx)
        ub = np.asarray(ub) if ub is not None else np.inf * np.ones(cost_model.nx)

        vg_funs, raw_funs, bounds = [], [], []
        for i in range(N):
            isi = cost_model.isi[i]

            def obj_i(xi, x, p, rho, i=i, isi=isi):
                x_full = x.at[isi].set(xi)
                fi = cost_model.costs(x_full, p, theta)[i]
                if ng > 0:
                    fi = fi + rho * jnp.sum(jnp.maximum(g(x_full, p), 0.0) ** 2)
                return fi

            vg_funs.append(jax.jit(jax.value_and_grad(obj_i, argnums=0)))
            raw_funs.append(jax.jit(
                lambda x, p, i=i: cost_model.costs(x, p, theta)[i]))
            bounds.append((lb[isi], ub[isi]))

        # Trigger the one-time trace/compile now, for every agent, so its
        # cost is attributed to compile_parametric_br() rather than
        # inflating the first best_response() call for each agent.
        x0 = jnp.zeros(cost_model.nx)
        p0 = jnp.zeros(npar)
        rho0 = jnp.asarray(1e5)
        for i in range(N):
            xi0 = jnp.zeros(cost_model.dims[i])
            val, grad = vg_funs[i](xi0, x0, p0, rho0)
            val.block_until_ready()
            grad.block_until_ready()
            raw_funs[i](x0, p0).block_until_ready()

        self._br = SimpleNamespace(theta=theta, vg_funs=vg_funs, raw_funs=raw_funs, bounds=bounds)
        return self

    def best_response(self, p, theta, i, x, x0=None, rho=1e5, tol=1e-8, maxiter=200):
        """Compute agent i's best response argmin_{x_i} J_i(x,p;theta) at
        the given joint strategy x and parameter p, honoring the shared/box
        constraints set at construction. Compiles the underlying
        value-and-gradient function ONCE per (instance, theta) -- via
        compile_parametric_br(theta), called automatically here if not
        already done -- and reuses that compiled executable for every
        subsequent call with the SAME theta object, regardless of i, x, p,
        rho, tol, or maxiter (none of which affect the compiled trace: rho
        is itself a traced argument, so varying it never triggers a
        recompile). Returns a SimpleNamespace (fields .x, the full decision
        vector with only agent i's slice updated, .f, .stats), matching
        nashopt.GNEP.best_response()'s shape.
        """
        self.compile_parametric_br(theta)
        cost_model = self.cost_model
        isi = cost_model.isi[i]
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        rho = jnp.asarray(rho)
        xi0 = x[isi] if x0 is None else jnp.asarray(x0)
        lb_i, ub_i = self._br.bounds[i]
        vg_fun = self._br.vg_funs[i]

        def fun(xi_np):
            val, grad = vg_fun(jnp.asarray(xi_np), x, p, rho)
            return float(val), np.asarray(grad, dtype=np.float64)

        t0 = time.perf_counter()
        res = minimize(fun, np.asarray(xi0), method='L-BFGS-B', jac=True,
                       bounds=list(zip(np.asarray(lb_i), np.asarray(ub_i))),
                       options=dict(maxiter=maxiter, maxfun=maxiter, ftol=tol, gtol=tol))
        elapsed = time.perf_counter() - t0

        x_new = np.asarray(x).copy()
        x_new[np.asarray(isi)] = res.x

        stats = SimpleNamespace(elapsed_time=elapsed, iters=res.nit, solver=res)
        sol = SimpleNamespace()
        sol.x = x_new
        sol.f = float(self._br.raw_funs[i](jnp.asarray(x_new), p))
        sol.stats = stats
        return sol

    # ------------------------------------------------------------------
    # Closed-form fast path for a pure quadratic (potential=None) cost model.
    # ------------------------------------------------------------------

    def solve_quadratic(self, p, theta, proximal=True, solver='daqp', **qp_opts):
        """Fast path for a pure quadratic StructuredMonotoneCost (potential=None).

        When no shared constraints are set, the game is an unconstrained
        quadratic Nash game, so its NE solves the pseudogradient linear
        system F(x,p) = A@x + q = 0 directly, rather than going through
        qp_gnep. Otherwise (shared linear/box constraints present), it still
        delegates to nashopt.lq.qp_gnep, matching the pattern in
        example_quad_game.py.
        """
        cost_model = self.cost_model
        if getattr(cost_model, "potential_spec", None) is not None:
            raise ValueError("solve_quadratic only supports potential=None cost models")

        theta_quad = theta[:cost_model.n_param_quad]
        A, q = cost_model.pseudogradient_matrix(p, theta_quad)
        A, q = np.array(A), np.array(q)

        unconstrained = all(self.constraints.get(k) is None for k in ('A', 'b', 'lb', 'ub'))
        if unconstrained:
            x = -np.linalg.solve(A, q)
            return SimpleNamespace(
                x=x, elapsed_time=0., status_str='converged', num_iters=1,
                info={'converged': True, 'final_gap': 0.})

        from nashopt.lq.qp_gnep import qp_gnep
        return qp_gnep(
            dim=cost_model.dims, Q=[A] * cost_model.N, c=[q] * cost_model.N,
            A=self.constraints.get('A'), b=self.constraints.get('b'),
            lb=self.constraints.get('lb'), ub=self.constraints.get('ub'),
            proximal=proximal, solver=solver, **qp_opts)

    # ------------------------------------------------------------------
    # Variational GNE via Korpelevich's extragradient method (mirrors
    # nashopt.nonlinear.nl_extragrad.extragrad_nlgnep, but with F(x,p;theta)
    # and the shared constraint's Jacobian JIT-compiled ONCE with p a
    # genuine traced argument -- see compile_parametric_extragrad()).
    # ------------------------------------------------------------------

    def compile_parametric_extragrad(self, theta):
        """JIT-compile ONCE the pseudogradient F(x,p;theta) =
        cost_model.pseudogradient(x,p,theta) and the shared constraint
        g(x,p) (and its Jacobian w.r.t. x), with p a genuine JAX-traced
        argument rather than a value baked into a Python closure -- exactly
        the same idea as compile_parametric(), applied to the extragradient
        solve instead of the KKT root-find. Idempotent: a repeat call with
        the SAME theta object is a no-op; solve_extragrad() calls this
        automatically on first use (or whenever theta changes), so calling
        it explicitly is only needed to report compile time separately from
        the first solve_extragrad() call. Returns self, so it can be
        chained, e.g.
        `eq_solver = EquilibriumSolver(cost_model, constraints).compile_parametric_extragrad(theta)`.
        """
        if self._extragrad is not None and self._extragrad.theta is theta:
            return self  # already compiled for this exact theta object

        if self._nonsmooth_potential():
            raise NotImplementedError(
                "Automatic epigraph/slack reformulation for nonsmooth (PWA/PWQ) "
                "potentials (Sec. 4.1) is not implemented for solve_extragrad "
                "either; solve manually.")

        cost_model = self.cost_model
        npar = cost_model.npar
        nx = cost_model.nx

        F_fn = jax.jit(lambda x, p: cost_model.pseudogradient(x, p, theta))
        g, ng = self._build_shared_g(npar)
        if ng > 0:
            g_fn = jax.jit(g)
            dg_fn = jax.jit(jax.jacobian(g, argnums=0))
        else:
            g_fn, dg_fn = None, None

        # Trigger the one-time trace/compile now, so its cost is attributed
        # to compile_parametric_extragrad() rather than inflating the first
        # solve_extragrad() call.
        x0 = jnp.zeros(nx)
        p0 = jnp.zeros(npar)
        F_fn(x0, p0).block_until_ready()
        if g_fn is not None:
            g_fn(x0, p0).block_until_ready()
            dg_fn(x0, p0).block_until_ready()

        lb = self.constraints.get('lb')
        ub = self.constraints.get('ub')
        lb = np.asarray(lb, dtype=np.float64) if lb is not None else -np.inf * np.ones(nx)
        ub = np.asarray(ub, dtype=np.float64) if ub is not None else np.inf * np.ones(nx)

        facade = _ExtragradConstraints(nx, lb, ub, ng, g_fn, dg_fn)

        self._extragrad = SimpleNamespace(
            theta=theta, F_fn=F_fn, facade=facade,
            proj_y=None, proj_x=None, proj_kind=None)
        return self

    def _extragrad_projectors(self, projection_solver, rho):
        """Build the IPOPT/TRF projectors onto the shared/box constraints
        ONCE (they read only the fixed problem dimensions/bounds off
        self._extragrad.facade at construction) and cache them on
        self._extragrad, reused for every subsequent solve_extragrad() call
        -- only facade.p and each project() call's target point v change
        per call, so no per-call rebuilding (IPOPT problem setup or JAX
        retracing) happens across a loop over many test points, unlike
        nashopt.GNEP.solve(solver='extragrad') which rebuilds a fresh GNEP
        (and hence fresh projectors) every time. Rebuilt only if
        projection_solver/rho change from the cached pair.
        """
        ex = self._extragrad
        kind = (projection_solver, rho)
        if ex.proj_y is not None and ex.proj_kind == kind:
            return ex.proj_y, ex.proj_x

        from nashopt.nonlinear.nl_extragrad import _Projector, _ProjectorTRF

        projection_solver = projection_solver.lower()
        if projection_solver == "ipopt":
            ex.proj_y = _Projector(ex.facade)
            ex.proj_x = _Projector(ex.facade)
        elif projection_solver == "trf":
            ex.proj_y = _ProjectorTRF(ex.facade, rho=rho)
            ex.proj_x = _ProjectorTRF(ex.facade, rho=rho)
        else:
            raise ValueError(f"Unknown projection_solver '{projection_solver}'. Use 'ipopt' or 'trf'.")
        ex.proj_kind = kind
        return ex.proj_y, ex.proj_x

    def solve_extragrad(self, p, theta, x0=None, alpha=None, tol=1e-8, maxiter=1000,
                        projection_solver='ipopt', rho=1e5, verbose=0):
        """Solve for x*(p): the variational GNE of cost_model.costs(.,.,theta)
        under the constraints set at construction, via Korpelevich's
        extragradient method

            y^k     = P_X(x^k - alpha*F(x^k))
            x^{k+1} = P_X(x^k - alpha*F(y^k))

        (same algorithm as nashopt.nonlinear.nl_extragrad.extragrad_nlgnep /
        nashopt.GNEP.solve(solver='extragrad')). Unlike calling
        nashopt.GNEP.solve(solver='extragrad') on a freshly-built GNEP,
        F(x,p;theta) and the shared constraint's Jacobian are JIT-compiled
        ONCE per (instance, theta) -- via compile_parametric_extragrad(theta),
        called automatically here if not already done -- with p a genuine
        traced argument, and the IPOPT/TRF projectors are likewise built
        ONCE and cached; every subsequent call with the SAME theta object
        (regardless of p) reuses all of these instead of retracing/
        recompiling and rebuilding the projectors from scratch, so a loop
        over many test points pays that setup cost only once.

        Parameters:
            p: parameter vector.
            theta: cost-model parameters (e.g. the fit result's theta).
            x0: initial point (default: zeros).
            alpha: step size. If None, estimated as 0.99/L, with L the
                Lipschitz constant of F approximated by a single finite-
                difference ratio (as extragrad_nlgnep's own default) -- pass
                e.g. 0.99/cost_model.lipschitz_constant(theta, p) (paper's
                Proposition 4.2) for a bound-based, cost-free-of-extra-
                F-evaluations alternative.
            tol: stop when ||x^{k+1}-x^k||_inf < tol.
            maxiter: maximum number of extragradient iterations.
            projection_solver: 'ipopt' (default, requires cyipopt) or 'trf'
                (scipy.optimize.least_squares, box bounds hard, other
                constraints penalized by rho).
            rho: penalty weight for the 'trf' projector (ignored for 'ipopt').
            verbose: 0 silent, >0 prints per-iteration residuals.

        Returns a SimpleNamespace (fields .x, .res, .lam (always []), .stats
        with .kkt_evals/.elapsed_time/.status_str/.info, .norm_residual),
        matching nashopt.nonlinear.nl_extragrad.solve_extragrad()'s shape.
        """
        self.compile_parametric_extragrad(theta)
        ex = self._extragrad
        nx = self.cost_model.nx
        p = jnp.asarray(p)

        x0_np = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

        def F_np(x_np):
            return np.asarray(ex.F_fn(jnp.asarray(x_np), p))

        if alpha is None:
            rng = np.random.default_rng(0)
            eps = 1e-4 * max(np.linalg.norm(x0_np), 1.0)
            xb = x0_np + eps * rng.standard_normal(nx)
            Fa, Fb = F_np(x0_np), F_np(xb)
            L = np.linalg.norm(Fa - Fb) / max(np.linalg.norm(x0_np - xb), 1e-15)
            alpha = 0.99 / max(L, 1e-12)

        proj_y, proj_x = self._extragrad_projectors(projection_solver, rho)
        ex.facade.p = p

        x = proj_x.project(x0_np)  # project x0 to feasibility before timing
        t_start = time.perf_counter()
        status_str = "max_iterations_reached"
        err = np.nan
        k = -1
        for k in range(maxiter):
            Fx = F_np(x)
            y = proj_y.project(x - alpha * Fx)
            Fy = F_np(y)
            x_new = proj_x.project(x - alpha * Fy)

            err = np.linalg.norm(x_new - x, np.inf)
            x = x_new

            if verbose:
                print(f"  extragrad iter {k + 1}: ||dx||_inf={err:.3e}")
            if err < tol:
                status_str = "converged"
                break
        elapsed = time.perf_counter() - t_start
        converged = status_str == "converged"

        if verbose > 0:
            color = "\033[1;32m" if converged else "\033[1;31m"
            status = "converged" if converged else "reached the maximum number of iterations"
            print(f"{color}Extragradient method {status} after {k + 1} iterations: "
                  f"||x^(k+1) - x^k||_inf = {err:.3e}, time = {elapsed:.3f} seconds.\033[0m")

        stats = SimpleNamespace(solver="extragrad", kkt_evals=k + 1,
                                elapsed_time=elapsed, status_str=status_str,
                                info={"converged": converged, "final_err": float(err)})
        return SimpleNamespace(x=x, res=np.atleast_1d(np.asarray(err)), lam=[],
                               stats=stats, norm_residual=float(err))
