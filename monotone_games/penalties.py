"""Penalty terms promoting monotonicity of a CostModel's pseudogradient, as described in [1].

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 

(C) 2026 A. Bemporad
"""

import abc

import jax
import jax.numpy as jnp


class MonotonicityPenalty(abc.ABC):
    """Loss term promoting monotonicity of a CostModel's pseudogradient.

    Evaluated on a monotonicity-sample set D_M = (Xm,Pm) that carries no cost
    or best-response labels, only decision/parameter values in the range of
    interest (e.g. the smallest hyperbox containing the training samples).
    """

    def __init__(self, gamma=1.e3):
        self.gamma = gamma

    def param_shapes(self, cost_model):
        """Extra theta shapes this penalty needs beyond cost_model.param_shapes().

        Default: none. Override (see AuxiliaryPotentialPenalty) when the
        penalty owns auxiliary trainable parameters.
        """
        return []

    @abc.abstractmethod
    def __call__(self, cost_model, Xm, Pm, theta, theta_extra=None):
        """Return the scalar penalty value evaluated over the batch (Xm,Pm)."""


class ViolationPenalty(MonotonicityPenalty):
    """L_M1 (eq. monotonicity-violation-1): penalizes pairwise violations of

        (x_j-x_h)^T(F(x_j,p_j)-F(x_h,p_j)) >= mu ||x_j-x_h||^2

    over all ordered pairs (j,h) in the sample set D_M=(Xm,Pm) (both samples
    taken at p_j, per the paper's eq.).
    """

    def __init__(self, gamma=1.e3, mu=0.):
        super().__init__(gamma)
        self.mu = mu

    def __call__(self, cost_model, Xm, Pm, theta, theta_extra=None):
        M = Xm.shape[0]

        def pair_violation(xj, pj, xh):
            Fj = cost_model.pseudogradient(xj, pj, theta)
            Fh = cost_model.pseudogradient(xh, pj, theta)
            lhs = self.mu * jnp.sum((xj - xh) ** 2)
            rhs = (xj - xh) @ (Fj - Fh)
            return jnp.maximum(0., lhs - rhs)

        def row(xj, pj):
            return jax.vmap(lambda xh: pair_violation(xj, pj, xh))(Xm)

        viol = jax.vmap(row)(Xm, Pm)  # (M,M)
        mask = 1. - jnp.eye(M)  # exclude the j==h diagonal
        return self.gamma / (M * (M - 1)) * jnp.sum((viol * mask) ** 2)


class JacobianEigPenalty(MonotonicityPenalty):
    """L_M2 (eq. monotonicity-violation-2): promotes mu-monotonicity by
    penalizing samples where the symmetric pseudogradient Jacobian is not
    bounded below by ``mu I``.

    Uses ``JF + JF.T`` (rather than its half) to preserve the original
    penalty scaling when mu=0, so its smallest eigenvalue must be at least
    ``2*mu``.
    """

    def __init__(self, gamma=1.e3, mu=0.):
        super().__init__(gamma)
        self.mu = mu

    def __call__(self, cost_model, Xm, Pm, theta, theta_extra=None):
        def sym_min_eig(x, p):
            JF = cost_model.pseudogradient_jacobian(x, p, theta)
            return jnp.linalg.eigvalsh(JF + JF.T)[0]

        lam_min = jax.vmap(sym_min_eig)(Xm, Pm)
        M = Xm.shape[0]
        return self.gamma / M * jnp.sum(jnp.maximum(0., 2. * self.mu - lam_min) ** 2)


class AuxiliaryPotentialPenalty(MonotonicityPenalty):
    """L_M3 (eq. monotonicity-violation-3): promotes mu-monotonicity by
    matching the symmetric pseudogradient Jacobian, after removing ``2*mu I``,
    to the Hessian of an auxiliary input-convex network Phi(x,p;theta_phi).
    The Hessian is positive semidefinite by construction. theta_phi's params
    are appended to theta during training (via param_shapes) and can be
    discarded after training.
    """

    def __init__(self, gamma=1.e3, phi_layers=(16, 16),
                 activation=jax.nn.softplus, mu=0.):
        super().__init__(gamma)
        self.phi_layers = list(phi_layers)
        self.activation = activation
        self.mu = mu

    def param_shapes(self, cost_model):
        nx, npar, layers = cost_model.nx, cost_model.npar, self.phi_layers
        shapes = [(layers[0], nx), (layers[0], npar), (layers[0],)]  # Wx0, Wp0, b0
        for l in range(len(layers) - 1):
            shapes += [(layers[l + 1], layers[l]), (layers[l + 1], nx),
                       (layers[l + 1], npar), (layers[l + 1],)]       # Wz, Wx, Wp, b
        shapes += [(1, layers[-1]), (1, nx), (1, npar), (1,)]          # output layer
        return shapes

    def _phi(self, x, p, theta_phi):
        act = self.activation
        Wx0, Wp0, b0 = theta_phi[:3]
        y = act(Wx0 @ x + Wp0 @ p + b0)
        idx = 3
        for l in range(len(self.phi_layers) - 1):
            Wz, Wx, Wp, b = theta_phi[idx:idx + 4]
            y = act(jax.nn.softplus(Wz) @ y + Wx @ x + Wp @ p + b)
            idx += 4
        Wz, Wx, Wp, b = theta_phi[idx:idx + 4]
        return (jax.nn.softplus(Wz) @ y + Wx @ x + Wp @ p + b)[0]

    def __call__(self, cost_model, Xm, Pm, theta, theta_extra=None):
        def violation(x, p):
            JF = cost_model.pseudogradient_jacobian(x, p, theta)
            H = jax.hessian(self._phi, argnums=0)(x, p, theta_extra)
            residual = JF + JF.T - 2. * self.mu * jnp.eye(cost_model.nx) - H
            return jnp.linalg.norm(residual, ord='fro')

        M = Xm.shape[0]
        v = jax.vmap(violation)(Xm, Pm)
        return self.gamma / M * jnp.sum(v ** 2)
