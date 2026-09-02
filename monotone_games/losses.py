"""Data-fit losses for learning a parametric game's costs, as described in [1].

[1] A. Bemporad and T. Tatarenko, "Learning Parametric Monotone Games,"
    arXiv preprint, 2026.

(C) 2026 A. Bemporad
"""

import abc

import jax
import jax.numpy as jnp
import numpy as np


class DataLoss(abc.ABC):
    """A fit term matching a CostModel against a dataset."""

    @abc.abstractmethod
    def __call__(self, cost_model, data, theta):
        """Return the scalar loss value for the given data batch."""


class CostSampleLoss(DataLoss):
    """L_DJ (eq. training_problem-J): direct MSE fit to cost samples
    D_J = (X,P,J), J[k,i] = tilde_J_i(x_k,p_k).

    GameLearner special-cases CostSampleLoss to reuse jax_sysid's built-in
    output-vs-target MSE machinery directly; this class's __call__ is
    provided for standalone use/testing and returns the same value.
    """

    def __call__(self, cost_model, data, theta):
        X, P, J = data

        def costs_row(x, p):
            return jnp.stack(cost_model.costs(x, p, theta))

        Jhat = jax.vmap(costs_row)(X, P)
        return jnp.sum((Jhat - J) ** 2) / X.shape[0]


class BestResponseLoss(DataLoss):
    """L_DX (eq. bilevel): inverse bilevel fit to best-response samples
    D_X = (X,P,agent_idx), where X[k,:] holds the observed decision vector of
    all agents at sample k and agent_idx[k] identifies which agent's slice
    x_{k,i} is the observed best response to x_{k,-i} (the other slices of
    X[k] provide the opponents' context x_{k,-i}).

    Requires a differentiable argmin_{x_i} J_i(x,p;theta). If cost_model
    exposes best_response_affine(i,x,p,theta) (closed form for quadratic
    costs, see inverse_quadratic.py / QuadraticInverseSolver.to_cost_model),
    that is used and the loss reduces to a plain least-squares fit; otherwise
    this default unrolls n_inner_steps of gradient descent on x_i (JAX
    autodiff differentiates through the unrolled loop w.r.t. theta) as an
    approximation to the true argmin. This unroll-vs-closed-form choice is
    not settled by the paper -- swap in a proper implicit-diff or
    differentiable-QP-layer best response (e.g. qpax/cvxpylayers, as noted
    in the paper for the constrained case) for higher fidelity.
    """

    def __init__(self, n_inner_steps=20, inner_lr=0.1):
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

    def best_response(self, cost_model, i, x, p, theta):
        """argmin_{x_i} J_i(x,p;theta), differentiable w.r.t. theta."""
        if hasattr(cost_model, "best_response_affine"):
            return cost_model.best_response_affine(i, x, p, theta)
        isi = cost_model.isi[i]
        xi0 = x[isi]

        def obj(xi):
            xx = x.at[isi].set(xi)
            return cost_model.costs(xx, p, theta)[i]

        xi = xi0
        for _ in range(self.n_inner_steps):
            xi = xi - self.inner_lr * jax.grad(obj)(xi)
        return xi

    def __call__(self, cost_model, data, theta):
        X, P, agent_idx = data
        # agent_idx must be a concrete array (it selects which per-agent slice
        # each residual lives on, so it cannot be a traced value); group the
        # samples per agent and vmap within each group.
        agent_idx = np.asarray(agent_idx)
        loss = 0.
        for i in range(cost_model.N):
            rows = np.where(agent_idx == i)[0]
            if rows.size == 0:
                continue

            def residual(x, p, i=i):
                xi_hat = self.best_response(cost_model, i, x, p, theta)
                return xi_hat - x[cost_model.isi[i]]

            r = jax.vmap(residual)(X[rows], P[rows])
            loss = loss + jnp.sum(r ** 2)
        return loss / X.shape[0]


class StationarityLoss(DataLoss):
    """NLS form of inverse learning that supports a potential term Psi
    (paper's eq. nls-nonquadratic): at an observed unconstrained best
    response x_{k,i_k}, the first-order optimality condition
    F_{i_k}(x_k,p_k;theta) = grad_{x_{i_k}} J_{i_k} = 0 holds, so the loss is
    the stationarity residual

        L(theta) = 1/K sum_k ||F_{i_k}(x_k,p_k;theta)||^2
                   + trace_penalty/K sum_k (tr(A(p_k;theta)) - n)^2

    This requires no inner argmin (no bilevel structure), at the price of an
    A_ii-type weighting of the best-response error: for a quadratic model
    without potential, F_i = A_ii(x_i - bar x_i), the weighted residual of
    eq. sdp-monotone; with a StructuredMonotoneCost the residual is
    A_ii x_i + A_{i,-i} x_{-i} + q_i + grad_{x_i} Psi(x,p), so the same
    formulation covers nonlinear monotone games with an input-convex
    potential and NN-parameterized C(p), D(p), q(p) (Aq={'type':'NN',...}).

    Unlike the scale-invariant argmin residual of eq. bilevel, this residual
    is homogeneous in theta's scale, so theta=0 is a trivial minimizer when
    mu=0; the trace term (trace_penalty=1 gives exactly the penalty in
    eq. nls-nonquadratic, the soft analog of eq. sdp-monotone's tr(A)=n
    constraint) rules it out. Applies only to cost models exposing
    pseudogradient_matrix.

    Data layout: (X, P, agent_idx), same as BestResponseLoss.
    """

    def __init__(self, trace_penalty=1.):
        self.trace_penalty = trace_penalty

    def __call__(self, cost_model, data, theta):
        X, P, agent_idx = data
        agent_idx = np.asarray(agent_idx)
        loss = 0.
        for i in range(cost_model.N):
            rows = np.where(agent_idx == i)[0]
            if rows.size == 0:
                continue

            def residual(x, p, i=i):
                g = jax.grad(lambda xx: cost_model.costs(xx, p, theta)[i])(x)
                return g[cost_model.isi[i]]

            r = jax.vmap(residual)(X[rows], P[rows])
            loss = loss + jnp.sum(r ** 2)
        loss = loss / X.shape[0]

        if self.trace_penalty > 0. and hasattr(cost_model, 'pseudogradient_matrix'):
            theta_quad = theta[:cost_model.n_param_quad]

            def trace_A(p):
                A, _ = cost_model.pseudogradient_matrix(p, theta_quad)
                return jnp.trace(A)

            tr = jax.vmap(trace_A)(P)
            loss = loss + self.trace_penalty * jnp.mean((tr - cost_model.nx) ** 2)
        return loss


class NESampleRegularizer:
    """Optional NE-sample regularizer (Sec. 2.3, eq. NE-samples-F/J/X): L_EF
    plus either L_EJ or L_EX, added on top of a DataLoss when a second
    dataset of NE samples D_E is available.

    data_ne = (Xe, Pe, target), where target is either the cost values
    Je[k,i] (match='cost', eq. NE-samples-J) or the NE decisions Xe_star
    itself (match='best_response', eq. NE-samples-X; here target==Xe, i.e.
    data_ne=(Xe,Pe,Xe) is the typical call).
    """

    def __init__(self, lam1=1., lam2=1., match="cost", best_response_loss=None):
        if match not in ("cost", "best_response"):
            raise ValueError("match must be 'cost' or 'best_response'")
        self.lam1 = lam1
        self.lam2 = lam2
        self.match = match
        self.best_response_loss = best_response_loss or BestResponseLoss()

    def __call__(self, cost_model, data_ne, theta):
        Xe, Pe, target = data_ne
        F = jax.vmap(cost_model.pseudogradient, in_axes=(0, 0, None))(Xe, Pe, theta)
        loss = self.lam2 * jnp.sum(F ** 2) / Xe.shape[0]

        if self.match == "cost":
            Je = target

            def costs_row(x, p):
                return jnp.stack(cost_model.costs(x, p, theta))

            Jhat = jax.vmap(costs_row)(Xe, Pe)
            loss = loss + self.lam1 * jnp.sum((Jhat - Je) ** 2) / Xe.shape[0]
        else:
            def br_row(x, p):
                return jnp.concatenate([
                    self.best_response_loss.best_response(cost_model, i, x, p, theta)
                    - x[cost_model.isi[i]]
                    for i in range(cost_model.N)
                ])

            r = jax.vmap(br_row)(Xe, Pe)
            loss = loss + self.lam1 * jnp.sum(r ** 2) / Xe.shape[0]
        return loss
