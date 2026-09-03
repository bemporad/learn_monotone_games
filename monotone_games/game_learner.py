"""Training orchestration for a CostModel against a DataLoss (paper's eq. learning-problem-point).

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 
    
(C) 2026 A. Bemporad
"""

import time
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.models import StaticModel, find_best_model
from jax_sysid.utils import compute_scores
from joblib import cpu_count

from .equilibrium import EquilibriumSolver
from .losses import CostSampleLoss, BestResponseLoss, StationarityLoss


class GameLearner:
    """Orchestrates training of a CostModel against a DataLoss, with optional
    MonotonicityPenalty and NESampleRegularizer add-ons:

        min_theta  L_D(theta) + L_M(theta) + L_E(theta) + rho/2 ||theta||^2

    where L_D is data_loss, L_M is monotonicity_penalty (only meaningful when
    cost_model.is_monotone_by_construction is False), and L_E is
    ne_regularizer (which itself includes the L_EF pseudogradient term).
    Wraps jax_sysid's StaticModel/find_best_model.

    jax_sysid's StaticModel.fit(Y,U) always adds a base MSE(output_fcn(U),Y)
    term on top of any custom_regularization -- it cannot be disabled. That
    base MSE is exactly L_DJ, so when data_loss is a CostSampleLoss,
    GameLearner reuses it directly (Y=J, U=(X,P)) instead of recomputing it
    inside custom_regularization. Any other DataLoss (e.g. BestResponseLoss)
    lacks that Y-vs-output_fcn(U) form, so output_fcn/Y are set to
    identically-matching zeros (the automatic MSE term then contributes
    nothing) and the entire loss is computed inside custom_regularization.
    """

    def __init__(self, cost_model, data_loss, monotonicity_penalty=None,
                 ne_regularizer=None, rho=1.e-8):
        self.cost_model = cost_model
        self.data_loss = data_loss
        self.monotonicity_penalty = monotonicity_penalty
        self.ne_regularizer = ne_regularizer
        self.rho = rho
        self._n_model_params = None
        if monotonicity_penalty is not None and cost_model.is_monotone_by_construction:
            warnings.warn(
                "monotonicity_penalty given but cost_model.is_monotone_by_construction "
                "is True; the penalty is redundant (the game is already monotone for "
                "any theta) and will only add training-time overhead.")
        if (isinstance(data_loss, (BestResponseLoss, StationarityLoss))
                and getattr(cost_model, "phi_spec", None) is not None):
            warnings.warn(
                "data_loss fits best-response/stationarity data but cost_model.phi_spec "
                "is not None; phi_i(x_{-i},p) does not depend on x_i, so it cannot affect "
                "this loss or its gradient (only rho/2||theta||^2 will pull its parameters "
                "toward 0) -- construct cost_model with phi=None to avoid the dead "
                "parameters and wasted computation.")

    def _param_shapes(self):
        shapes = list(self.cost_model.param_shapes())
        self._n_model_params = len(shapes)
        if self.monotonicity_penalty is not None:
            shapes += self.monotonicity_penalty.param_shapes(self.cost_model)
        return shapes

    def _split_theta(self, theta_full):
        return theta_full[:self._n_model_params], theta_full[self._n_model_params:]

    def fit(self, data, data_m=None, data_ne=None, adam_epochs=1000,
            lbfgs_epochs=5000, seeds=0, val_data=None):
        """Fit the cost model.

        data: the dataset expected by self.data_loss, e.g. (X,P,J) for
            CostSampleLoss or (X,P,agent_idx) for BestResponseLoss.
        data_m: (Xm,Pm) monotonicity-sample set D_M, required if
            monotonicity_penalty is set.
        data_ne: (Xe,Pe,target) optional NE-sample set D_E, used if
            ne_regularizer is set.
        val_data: optional (X_val,P_val,J_val) used to select the best model
            across seeds when data_loss is a CostSampleLoss; ignored
            otherwise (see model-selection note below).

        Returns a SimpleNamespace(cost_model, theta, model, fit_score, training_time).
        """
        cost_model = self.cost_model
        nx, npar = cost_model.nx, cost_model.npar

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)

        param_shapes = self._param_shapes()
        is_direct = isinstance(self.data_loss, CostSampleLoss)

        if is_direct:
            X, P, J = data
            ny = cost_model.N

            @jax.jit
            def output_fcn(xp, theta_full):
                theta, _ = self._split_theta(theta_full)
                x, p = xp[:, :nx], xp[:, nx:]

                def costs_row(x, p):
                    return jnp.stack(cost_model.costs(x, p, theta))
                return jax.vmap(costs_row)(x, p)

            XP, Y = np.hstack((X, P)), np.asarray(J)
        else:
            ny = 1
            M = data[0].shape[0]

            @jax.jit
            def output_fcn(xp, theta_full):
                return jnp.zeros((xp.shape[0], 1))

            XP, Y = np.zeros((M, nx + npar)), np.zeros((M, 1))

        model = StaticModel(ny, nx, output_fcn)

        def init_fcn(seed):
            rng = np.random.RandomState(seed)
            return [rng.randn(*shape) for shape in param_shapes]

        model.init(params=init_fcn(0))

        @jax.jit
        def custom_regularization(theta_full):
            theta, theta_extra = self._split_theta(theta_full)
            loss = 0. if is_direct else self.data_loss(cost_model, data, theta)
            if self.monotonicity_penalty is not None and data_m is not None:
                Xm, Pm = data_m
                loss = loss + self.monotonicity_penalty(cost_model, Xm, Pm, theta, theta_extra)
            if self.ne_regularizer is not None and data_ne is not None:
                loss = loss + self.ne_regularizer(cost_model, data_ne, theta)
            return loss

        # jax_sysid applies rho_th*||theta||^2; the paper's eq. (learning-problem-point)
        # specifies rho/2 ||theta||^2, hence the factor 1/2 here
        model.loss(rho_th=self.rho / 2., custom_regularization=custom_regularization)
        model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        if not isinstance(seeds, list):
            if isinstance(seeds, (int, float)):
                seeds = [seeds]
            elif isinstance(seeds, np.ndarray):
                seeds = seeds.tolist()
            else:
                raise ValueError("seeds must be an int, float, list, or numpy array")

        if val_data is not None and is_direct:
            X_val, P_val, J_val = val_data
            XP_val, Y_val = np.hstack((X_val, P_val)), np.asarray(J_val)
        else:
            XP_val, Y_val = XP, Y

        t0 = time.time()
        if len(seeds) > 1:
            models = model.parallel_fit(Y, XP, init_fcn, seeds=seeds, n_jobs=cpu_count())
            if is_direct:
                model, _ = find_best_model(models, Y_val, XP_val, 'r2')
            else:
                # R2 against the dummy zero target is meaningless here; select
                # the seed that minimizes the actual training objective instead.
                losses = [float(custom_regularization(m.params)) for m in models]
                model = models[int(np.argmin(losses))]
        else:
            model.fit(Y, XP)
        training_time = time.time() - t0

        if is_direct:
            Jhat = model.predict(XP)
            fit_score, _, msg = compute_scores(Y, Jhat, fit='r2')
            print(msg)
        else:
            fit_score = None

        theta_full = model.params
        theta, _ = self._split_theta(theta_full)

        def equilibrium_solver(constraints=None, precompile=True):
            """Build an EquilibriumSolver for this fitted game (cost_model
            and theta fixed at this result's values); constraints (box
            bounds 'lb'/'ub', shared inequality 'g'/'ng' or 'A'/'b'/'S' --
            see EquilibriumSolver's docstring) are a property of the game/
            problem, not of training, so must be supplied here. If
            precompile (default True), eagerly compiles the joint-GNE KKT
            path now (compile_parametric(theta)) so its cost is attributed
            to this call rather than to the first solve(); the
            best-response path (compile_parametric_br) is left to compile
            lazily on the first best_response() call, since not every
            caller needs it. The returned EquilibriumSolver's solve()/
            best_response() are then ready to call directly, at only a
            root-find's/L-BFGS-B's cost each, for as many different p's as
            needed -- e.g. `eq = result.equilibrium_solver(constraints);
            eq.solve(p, result.theta, x0=...)`.
            """
            eq_solver = EquilibriumSolver(cost_model, constraints)
            if precompile:
                eq_solver.compile_parametric(theta)
            return eq_solver

        result = SimpleNamespace(cost_model=cost_model, theta=theta, model=model,
                                 fit_score=fit_score, training_time=training_time)
        result.equilibrium_solver = equilibrium_solver
        return result

    def monotonicity_check(self, theta, xmin, xmax, pmin, pmax, **globopt_opts):
        """Certify (mu-strong) monotonicity of the fitted game over a box.

        Globally minimizes the smallest eigenvalue of the symmetric part of
        the pseudogradient Jacobian over xmin <= x <= xmax, pmin <= p <= pmax
        via maxfit.solve_global_optimization; see CostModel.monotonicity_check
        for details and the accepted globopt_opts (mu, global_optimizer,
        ftol_abs, maxeval, xtol_rel).
        """
        return self.cost_model.monotonicity_check(theta, xmin, xmax, pmin, pmax,
                                                  **globopt_opts)

    def predict(self, model_out, X, P):
        """Predict costs J_i(x,p;theta) for the fitted model_out (CostSampleLoss models only)."""
        XP = np.hstack((X, P))
        return model_out.model.predict(XP)
