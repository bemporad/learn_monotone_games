"""Parametric cost models J_i(x,p;theta), i=1,...,N, for a parametric game.

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 
    
(C) 2026 A. Bemporad
"""

import abc
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


class CostModel(abc.ABC):
    """Base class for parametric game cost models J_i(x,p;theta).

    Subclasses represent the agents' cost functions and must be JAX-traceable:
    all methods below are called under jax.jit/vmap by GameLearner and
    EquilibriumSolver.
    """

    is_monotone_by_construction = False

    def __init__(self, dims, npar):
        self.dims = list(dims)
        self.N = len(dims)
        self.nx = int(sum(dims))
        self.npar = npar
        i1 = np.concatenate((np.array([0]), np.cumsum(self.dims[:-1])))
        i2 = np.cumsum(self.dims) - 1
        self.isi = [np.arange(i1[i], i2[i] + 1) for i in range(self.N)]
        self.nisi = [np.concatenate((np.arange(0, i1[i]), np.arange(i2[i] + 1, self.nx)))
                     for i in range(self.N)]

    @abc.abstractmethod
    def param_shapes(self):
        """Return the list of theta-array shapes defining this cost model."""

    def init_params(self, seed):
        """Default random init: one np.random.randn array per param_shapes() entry."""
        rng = np.random.RandomState(seed)
        return [rng.randn(*shape) for shape in self.param_shapes()]

    @abc.abstractmethod
    def costs(self, x, p, theta):
        """Return the list of N scalar costs J_i(x,p;theta). Must be JAX-traceable."""

    def pseudogradient(self, x, p, theta):
        """F(x,p;theta) = col(dJ_1/dx_1,...,dJ_N/dx_N). Generic autodiff; override for a closed form."""
        dJ_dx = jax.jacfwd(self.costs, argnums=0)(x, p, theta)
        return jnp.concatenate([dJ_dx[i][self.isi[i]] for i in range(self.N)])

    def pseudogradient_jacobian(self, x, p, theta):
        """Jacobian of the pseudogradient w.r.t. x. Generic autodiff."""
        return jax.jacfwd(self.pseudogradient, argnums=0)(x, p, theta)

    def monotonicity_check(self, theta, xmin, xmax, pmin, pmax, mu=0.,
                           global_optimizer='direct', ftol_abs=1e-8,
                           maxeval=2000, xtol_rel=1e-5):
        """Certify (mu-strong) monotonicity of the game over a box by global optimization.

        Minimizes lambda_min(.5*(JF(x,p;theta) + JF(x,p;theta)^T)), the smallest
        eigenvalue of the symmetric part of the pseudogradient Jacobian, over
        the box xmin <= x <= xmax, pmin <= p <= pmax, using
        maxfit.solve_global_optimization.

        Parameters:
            theta: cost-model parameters (e.g. the fit result's theta).
            xmin, xmax: box bounds on x, arrays of length nx.
            pmin, pmax: box bounds on p, arrays of length npar.
            mu: strong-monotonicity threshold used for the is_monotone flag.
            global_optimizer, ftol_abs, maxeval, xtol_rel: options forwarded
                to maxfit.solve_global_optimization.

        Returns a SimpleNamespace with:
            lam_min: smallest eigenvalue found over the box (global minimum,
                up to solver tolerances).
            x, p: minimizer at which lam_min is attained.
            is_monotone: True if lam_min >= mu.
        """
        from maxfit import solve_global_optimization

        theta = [np.asarray(th) for th in theta]
        nx = self.nx

        @jax.jit
        def min_eig(z):
            x, p = z[:nx], z[nx:]
            JF = self.pseudogradient_jacobian(x, p, theta)
            return jnp.linalg.eigvalsh(.5 * (JF + JF.T))[0]

        def globopt_loss(z):
            # solve_global_optimization evaluates this in joblib worker
            # processes, which do not inherit the parent's jax config
            if not jax.config.jax_enable_x64:
                jax.config.update("jax_enable_x64", True)
            return float(min_eig(jnp.asarray(z)))

        lb = np.concatenate((np.asarray(xmin, dtype=float).reshape(-1),
                             np.asarray(pmin, dtype=float).reshape(-1)))
        ub = np.concatenate((np.asarray(xmax, dtype=float).reshape(-1),
                             np.asarray(pmax, dtype=float).reshape(-1)))
        if lb.size != nx + self.npar or ub.size != nx + self.npar:
            raise ValueError(f"box bounds must have sizes nx={nx} (x) and npar={self.npar} (p)")

        z_opt, lam_min = solve_global_optimization(
            globopt_loss, lb, ub, global_optimizer=global_optimizer,
            ftol_abs=ftol_abs, maxeval=maxeval, xtol_rel=xtol_rel)
        z_opt = np.asarray(z_opt)
        return SimpleNamespace(lam_min=lam_min, x=z_opt[:nx], p=z_opt[nx:],
                               is_monotone=bool(lam_min >= mu))

    def best_response(self, i, x, p, theta, **solver_opts):
        """argmin_{x_i} J_i(x,p;theta), holding x_{-i} fixed.

        Default: delegate to nashopt.GNEP.best_response on the full game.
        Subclasses with a closed-form best response (e.g. quadratic costs,
        see inverse_quadratic.py) should override for speed and exact
        differentiability.
        """
        from nashopt import GNEP
        f = [lambda xx, j=j: self.costs(xx, p, theta)[j] for j in range(self.N)]
        gnep = GNEP(sizes=self.dims, f=f)
        return gnep.best_response(i, x, **solver_opts)


class StructuredMonotoneCost(CostModel):
    """Monotonicity-by-construction cost model (paper's Lemma 4.1/4.2).

    Pseudogradient F(x,p;theta) = (C^T C + mu I + D - D^T) x + q + grad_x Psi(x,p;theta),
    with C upper-triangular, D block strictly upper-triangular (so C^T C + mu I
    is symmetric PSD and D - D^T is skew-symmetric), and Psi an optional convex
    potential selected by `potential`:
        None: no potential term (pure quadratic game, Lemma 2)
        {'type': 'PWA', 'L': L}: convex piecewise-affine potential, L pieces
        {'type': 'PWQ', 'L': L}: convex piecewise-quadratic potential, L pieces
        {'type': 'NN', 'layers': layers, 'activation': act_fun}: input-convex
            neural network potential with the given layer sizes and a convex
            nondecreasing activation (e.g. relu, softplus, elu)
    The agent-specific terms phi_i(x_{-i},p) -- arbitrary in Lemma 1's cost
    form since they do not depend on x_i and hence do not affect the
    pseudogradient nor monotonicity -- are selected by `phi`:
        None: no phi terms
        'quadratic' (default): quadratic in x_{-i} with p-affine coefficients
            (paper's eq. Ji-quadratic normalization, generalized to
            parametric coefficients)
        {'type': 'NN', 'layers': layers, 'activation': act_fun}: generic
            per-agent neural networks in x_{-i} whose weights are affine in p
            (same hypernetwork style as the potential net: p enters through
            the weights, never through the activations, so no p-rescaling is
            needed)
    The resulting game is monotone (mu-strongly monotone if mu>0) for every
    theta and every p, by construction (Lemma 4.1).
    """

    is_monotone_by_construction = True

    def __init__(self, dims, npar, mu=0., potential=None, phi='quadratic', Aq='constant'):
        """Aq selects the parameterization of the nonzero entries of C(p),
        D(p), q(p), always such that A(p) = C(p)^T C(p) + mu I + D(p) - D(p)^T
        is monotone for every p and theta:
            'affine': entries affine in p:
                C(p) = C_th1 p + C_th2, D(p) = D_th1 p + D_th2, q(p) = q1 p + q2
            'constant' (default): C, D constant (independent of p), q affine
                in p: C(p) = C_th2, D(p) = D_th2, q(p) = q1 p + q2
            {'type': 'NN', 'layers': layers, 'activation': act_fun}: a single
                MLP p -> [C entries, D entries, q] (feeding p through
                activations: rescale p to O(1) for good conditioning)
        """
        super().__init__(dims, npar)
        self.mu = mu
        self.potential_spec = potential
        if isinstance(phi, str):
            phi = {'type': phi}
        self.phi_spec = phi
        if isinstance(Aq, str):
            Aq = {'type': Aq}
        self.Aq_spec = Aq
        self.mask_C = np.triu(np.ones((self.nx, self.nx), dtype=bool))
        block_id = np.repeat(np.arange(self.N), self.dims)
        self.block_mask_D = block_id[:, None] < block_id[None, :]
        self._idx_C = np.where(self.mask_C)
        self._idx_D = np.where(self.block_mask_D)
        self._nC = int(self.mask_C.sum())
        self._nD = int(self.block_mask_D.sum())

        nx, npar_ = self.nx, self.npar
        match self.Aq_spec['type']:
            case 'affine':
                # C_th1, C_th2, D_th1, D_th2, q1, q2
                self._dims_Aq = [
                    (nx, nx, npar_), (nx, nx),
                    (nx, nx, npar_), (nx, nx),
                    (nx, npar_), (nx,),
                ]
            case 'constant':
                # C_th2, D_th2, q1, q2
                self._dims_Aq = [
                    (nx, nx),
                    (nx, nx),
                    (nx, npar_), (nx,),
                ]
            case 'NN':
                layers = self.Aq_spec['layers']
                n_out = self._nC + self._nD + nx
                self._dims_Aq = [(layers[0], npar_), (layers[0],)]
                for l in range(len(layers) - 1):
                    self._dims_Aq += [(layers[l + 1], layers[l]), (layers[l + 1],)]
                self._dims_Aq += [(n_out, layers[-1]), (n_out,)]
            case _:
                raise ValueError(f"Unknown Aq type {self.Aq_spec['type']!r}")
        # phi_i(x_{-i},p) parameters
        self._dims_phi = []
        self._phi_slices = []
        if phi is not None:
            match phi['type']:
                case 'quadratic':
                    # phi_Q, phi_c, phi_lin1, phi_lin2, phi_h
                    self._dims_phi = [(self.N, nx, nx), (self.N, nx, npar_),
                                       (self.N, nx, npar_), (self.N, nx), (self.N,)]
                case 'NN':
                    off = len(self._dims_Aq)
                    for i in range(self.N):
                        s = self._phi_param_shapes(i)
                        self._phi_slices.append((off, off + len(s)))
                        self._dims_phi += s
                        off += len(s)
                case _:
                    raise ValueError(f"Unknown phi type {phi['type']!r}")
        self.n_param_quad = len(self._dims_Aq) + len(self._dims_phi)
        self._dims_potential = self._potential_param_shapes()

    # -- potential ----------------------------------------------------------
    def _potential_param_shapes(self):
        nx, npar = self.nx, self.npar
        spec = self.potential_spec
        if spec is None:
            return []
        match spec["type"]:
            case "PWA":
                L = spec["L"]
                return [(L, nx, npar), (L, nx), (L, npar), (L,)]
            case "PWQ":
                L = spec["L"]
                return [(L, nx, nx, npar), (L, nx, nx), (L, nx, npar), (L, nx), (L, npar), (L,)]
            case "NN":
                layers = spec["layers"]
                shapes = [(layers[0], nx, npar), (layers[0], nx), (layers[0], npar), (layers[0],)]
                for i in range(len(layers) - 1):
                    shapes += [
                        (layers[i + 1], layers[i], npar), (layers[i + 1], layers[i]),
                        (layers[i + 1], nx, npar), (layers[i + 1], nx),
                        (layers[i + 1], npar), (layers[i + 1],),
                    ]
                shapes += [(1, layers[-1], npar), (1, layers[-1]), (1, nx, npar), (1, nx), (1, npar), (1,)]
                return shapes
        raise ValueError(f"Unknown potential type {spec['type']!r}")

    def potential(self, x, p, theta_pot):
        """Psi(x,p;theta_pot), the optional convex potential term (0. if potential=None)."""
        spec = self.potential_spec
        if spec is None:
            return 0.
        match spec["type"]:
            case "PWA":
                A1, A2, b1, b2 = theta_pot
                return jnp.max((A1 @ p + A2) @ x + (b1 @ p + b2))
            case "PWQ":
                C1, C2, A1, A2, b1, b2 = theta_pot
                Cp = C1 @ p + C2
                CpT = jnp.swapaxes(Cp, -1, -2)
                return jnp.max(.5 * x @ (CpT @ Cp) @ x + (A1 @ p + A2) @ x + (b1 @ p + b2))
            case "NN":
                layers = spec["layers"]
                act = spec["activation"]
                y = None
                for i in range(len(layers) + 1):
                    if i == 0:
                        W1, W2, b1, b2 = theta_pot[:4]
                        y = (W1 @ p + W2) @ x + b1 @ p + b2
                    else:
                        V1, V2, W1, W2, b1, b2 = theta_pot[6 * i - 2:6 * i + 4]
                        y = jax.nn.softplus(V1 @ p + V2) @ y + (W1 @ p + W2) @ x + b1 @ p + b2
                    if i < len(layers):
                        y = act(y)
                return y[0]

    # -- pseudogradient matrix ------------------------------------------------
    def pseudogradient_matrix(self, p, theta_quad):
        """A(p;theta) = C^T C + mu I + D - D^T (eq. pseudogradient-matrix), and q(p;theta)."""
        if self.Aq_spec['type'] == 'affine':
            C_th1, C_th2, D_th1, D_th2, q1, q2 = theta_quad[:6]
            C = jnp.where(self.mask_C, C_th1 @ p + C_th2, 0.)
            D = jnp.where(self.block_mask_D, D_th1 @ p + D_th2, 0.)
            q = q1 @ p + q2
        elif self.Aq_spec['type'] == 'constant':
            C_th2, D_th2, q1, q2 = theta_quad[:4]
            C = jnp.where(self.mask_C, C_th2, 0.)
            D = jnp.where(self.block_mask_D, D_th2, 0.)
            q = q1 @ p + q2
        else:  # 'NN': single MLP p -> [C entries, D entries, q]
            th = theta_quad[:len(self._dims_Aq)]
            act = self.Aq_spec['activation']
            y = act(th[0] @ p + th[1])
            idx = 2
            for l in range(len(self.Aq_spec['layers']) - 1):
                y = act(th[idx] @ y + th[idx + 1])
                idx += 2
            out = th[idx] @ y + th[idx + 1]
            C = jnp.zeros((self.nx, self.nx)).at[self._idx_C].set(out[:self._nC])
            D = jnp.zeros((self.nx, self.nx)).at[self._idx_D].set(out[self._nC:self._nC + self._nD])
            q = out[self._nC + self._nD:]
        A = C.T @ C + self.mu * jnp.eye(self.nx) + D - D.T
        return A, q

    def lipschitz_constant(self, theta, p=None):
        """Estimate the Lipschitz constant of the pseudogradient F(x,p;theta) (paper's
        Proposition~4.2).

        If the game is quadratic (potential=None) and A does not depend on p
        (Aq='constant'), p is ignored and the largest eigenvalue of the symmetric
        part of the (constant) pseudogradient matrix A(theta) is returned, since then
        F(x)=Ax+q and its Jacobian is A itself.

        Otherwise (Aq='affine', Aq a neural network, or a potential term is present)
        p must be given, and the largest eigenvalue of the symmetric part of
        A(p;theta) is evaluated at that p.

        If a potential term is present and it is an input-convex neural network
        (potential={'type': 'NN', ...}) with softplus activation, Proposition 4.2's
        bound L_Psi <= ||w_L||_2 H_L on the potential's contribution to the
        pseudogradient's Lipschitz constant is added to the eigenvalue above. Any
        other potential type, or an NN potential with a non-softplus activation, is
        not covered by Proposition 4.2 and an error message string is returned
        instead (no bound is computed).

        Parameters:
            theta: cost-model parameters (e.g. the fit result's theta).
            p: parameter vector; required unless the game is quadratic
                (potential=None) with Aq='constant'.

        Returns:
            A float (the estimated Lipschitz constant), or a string error message
            if the potential term is not an input-convex NN with softplus
            activation.
        """
        theta_quad = theta[:self.n_param_quad]
        theta_pot = theta[self.n_param_quad:]
        no_potential = self.potential_spec is None
        ignore_p = no_potential and self.Aq_spec['type'] == 'constant'
        if ignore_p:
            p_eval = jnp.zeros(self.npar)
        else:
            if p is None:
                raise ValueError("p cannot be None unless the game is quadratic "
                                 "(potential=None) with Aq='constant'")
            p_eval = jnp.asarray(p)

        A, _ = self.pseudogradient_matrix(p_eval, theta_quad)
        L = float(jnp.linalg.eigvalsh(.5 * (A + A.T))[-1])

        if not no_potential:
            spec = self.potential_spec
            if spec['type'] != 'NN':
                return (f"lipschitz_constant: Proposition 4.2 only covers an "
                        f"input-convex NN potential, not implemented yet for "
                        f"potential type {spec['type']!r}")
            if spec['activation'] is not jax.nn.softplus:
                return ("lipschitz_constant: Proposition 4.2's bound has not been "
                        "implemented yet for non-softplus activations")
            L += float(self._potential_lipschitz_increment(p_eval, theta_pot))
        return L

    @partial(jax.jit, static_argnums=(0,))
    def _potential_lipschitz_increment(self, p, theta_pot):
        """Proposition 4.2's bound L_Psi <= ||w_L||_2 H_L on the Lipschitz constant
        of grad(Psi) for an input-convex NN potential (potential={'type': 'NN', ...})
        with softplus activation and linear bypass. Bias terms do not enter the
        recursion since they do not affect the gradient's Lipschitz constant.
        jax.jit'd with self static (layers/potential_spec are fixed at
        construction; only p, theta_pot are traced), so repeated calls on the
        same instance (e.g. once per test p in a loop) reuse one compiled
        executable instead of retracing every time.
        """
        layers = self.potential_spec['layers']
        Lnn = len(layers)
        W1, W2 = theta_pot[0], theta_pot[1]
        W0 = W1 @ p + W2
        G = jnp.linalg.norm(W0, ord=2)
        H = .25 * G ** 2
        idx = 4
        for _ in range(1, Lnn):
            V1, V2, W1, W2 = theta_pot[idx], theta_pot[idx + 1], theta_pot[idx + 2], theta_pot[idx + 3]
            idx += 6
            Wk = jax.nn.softplus(V1 @ p + V2)
            Uk = W1 @ p + W2
            normWk = jnp.linalg.norm(Wk, ord=2)
            normUk = jnp.linalg.norm(Uk, ord=2)
            G = normWk * G + normUk
            H = .25 * G ** 2 + normWk * H
        V1, V2, W1, W2 = theta_pot[idx], theta_pot[idx + 1], theta_pot[idx + 2], theta_pot[idx + 3]
        wL = jax.nn.softplus(V1 @ p + V2).reshape(-1)
        return jnp.linalg.norm(wL) * H

    # -- agent-specific terms phi_i(x_{-i},p) ------------------------------------
    def _phi_param_shapes(self, i):
        # NN type: weights affine in p (hypernetwork style, like the potential net)
        layers = self.phi_spec['layers']
        n_mi, npar = self.nx - self.dims[i], self.npar
        shapes = [(layers[0], n_mi, npar), (layers[0], n_mi), (layers[0], npar), (layers[0],)]
        for l in range(len(layers) - 1):
            shapes += [(layers[l + 1], layers[l], npar), (layers[l + 1], layers[l]),
                       (layers[l + 1], n_mi, npar), (layers[l + 1], n_mi),
                       (layers[l + 1], npar), (layers[l + 1],)]
        shapes += [(1, layers[-1], npar), (1, layers[-1]), (1, n_mi, npar), (1, n_mi), (1, npar), (1,)]
        return shapes

    def _phi_nn(self, x_mi, p, theta_phi):
        layers = self.phi_spec['layers']
        act = self.phi_spec['activation']
        y = None
        for i in range(len(layers) + 1):
            if i == 0:
                W1, W2, b1, b2 = theta_phi[:4]
                y = (W1 @ p + W2) @ x_mi + b1 @ p + b2
            else:
                V1, V2, W1, W2, b1, b2 = theta_phi[6 * i - 2:6 * i + 4]
                y = (V1 @ p + V2) @ y + (W1 @ p + W2) @ x_mi + b1 @ p + b2
            if i < len(layers):
                y = act(y)
        return y[0]

    # -- costs ------------------------------------------------------------
    def param_shapes(self):
        return self._dims_Aq + self._dims_phi + self._dims_potential

    def _quadratic_costs(self, x, p, theta_quad):
        A, q = self.pseudogradient_matrix(p, theta_quad)
        isi, nisi = self.isi, self.nisi
        is_phi_quadratic = self.phi_spec is not None and self.phi_spec['type'] == 'quadratic'
        is_phi_nn = self.phi_spec is not None and self.phi_spec['type'] == 'NN'
        n_Aq = len(self._dims_Aq)
        if is_phi_quadratic:
            phi_Q, phi_c, phi_lin1, phi_lin2, phi_h = theta_quad[n_Aq:n_Aq + 5]
            phi_lin = phi_lin1 @ p + phi_lin2
        J = []
        for i in range(self.N):
            Ji = (.5 * x[isi[i]] @ A[isi[i], :][:, isi[i]] @ x[isi[i]]
                  + x[isi[i]] @ A[isi[i], :][:, nisi[i]] @ x[nisi[i]]
                  + q[isi[i]] @ x[isi[i]])
            if is_phi_quadratic:
                Ji = Ji + (x[nisi[i]] @ (phi_Q[i][nisi[i], :][:, nisi[i]] @ x[nisi[i]]
                                          + phi_c[i][nisi[i]] @ p)
                           + phi_lin[i][nisi[i]] @ x[nisi[i]] + phi_h[i])
            elif is_phi_nn:
                s0, s1 = self._phi_slices[i]
                Ji = Ji + self._phi_nn(x[nisi[i]], p, theta_quad[s0:s1])
            J.append(Ji)
        return J

    def costs(self, x, p, theta):
        theta_quad = theta[:self.n_param_quad]
        theta_pot = theta[self.n_param_quad:]
        J = self._quadratic_costs(x, p, theta_quad)
        Phi = self.potential(x, p, theta_pot)
        return [Ji + Phi for Ji in J]


class PenalizedConvexCost(CostModel):
    """Generic input-convex-in-x_i cost model (paper's Sec. 3).

    Each agent's cost is

        J_i(x,p;theta) = h_i(x_i,p;theta_own) + L_i(x_{-i},p;theta_couple)^T x_i
                         + m_i(x_{-i},p;theta_couple)

    where h_i is an input-convex neural network in x_i alone (guaranteeing the
    baseline per-agent convexity assumed throughout the paper) and L_i, m_i
    are generic MLPs of (x_{-i},p) only, so the coupling term is affine in
    x_i for fixed x_{-i},p and J_i stays convex in x_i (sum of a convex and
    an affine function of x_i) no matter what L_i, m_i are.

    Monotonicity of the resulting game is NOT guaranteed by construction; it
    must be promoted at training time with a MonotonicityPenalty (see
    penalties.py). This is one architecture choice among several possible
    generic input-convex parameterizations (not dictated by the paper);
    own_layers and coupling_layers size the two nets per agent.
    """

    is_monotone_by_construction = False

    def __init__(self, dims, npar, own_layers=(16, 16), coupling_layers=(16, 16),
                 activation=jax.nn.softplus):
        super().__init__(dims, npar)
        self.own_layers = list(own_layers)
        self.coupling_layers = list(coupling_layers)
        self.activation = activation
        self._own_slices = []
        self._couple_slices = []

    # -- own-variable input-convex net h_i(x_i,p;theta) ------------------------
    def _own_param_shapes(self, ni):
        npar, layers = self.npar, self.own_layers
        shapes = [(layers[0], ni), (layers[0], npar), (layers[0],)]  # Wx0, Wp0, b0
        for l in range(len(layers) - 1):
            shapes += [(layers[l + 1], layers[l]), (layers[l + 1], ni),
                       (layers[l + 1], npar), (layers[l + 1],)]       # Wz, Wx, Wp, b
        shapes += [(1, layers[-1]), (1, ni), (1, npar), (1,)]          # output layer
        return shapes

    def _own_cost(self, xi, p, theta_own):
        layers, act = self.own_layers, self.activation
        Wx0, Wp0, b0 = theta_own[:3]
        y = act(Wx0 @ xi + Wp0 @ p + b0)
        idx = 3
        for l in range(len(layers) - 1):
            Wz, Wx, Wp, b = theta_own[idx:idx + 4]
            y = act(jax.nn.softplus(Wz) @ y + Wx @ xi + Wp @ p + b)
            idx += 4
        Wz, Wx, Wp, b = theta_own[idx:idx + 4]
        return (jax.nn.softplus(Wz) @ y + Wx @ xi + Wp @ p + b)[0]

    # -- coupling nets L_i(x_{-i},p;theta), m_i(x_{-i},p;theta) -----------------
    # (arbitrary in x_{-i},p; only ever combined affinely with x_i, see class
    # docstring, so J_i's convexity in x_i is never at risk)
    def _coupling_param_shapes(self, i):
        npar, layers = self.npar, self.coupling_layers
        n_mi = self.nx - self.dims[i]
        n_out = self.dims[i] + 1  # L_i (dims[i],) followed by m_i (scalar)
        shapes = [(layers[0], n_mi + npar), (layers[0],)]
        for l in range(len(layers) - 1):
            shapes += [(layers[l + 1], layers[l]), (layers[l + 1],)]
        shapes += [(n_out, layers[-1]), (n_out,)]
        return shapes

    def _coupling_cost(self, xi, x_mi, p, theta_couple):
        act = self.activation
        z = jnp.concatenate([x_mi, p])
        W, b = theta_couple[0], theta_couple[1]
        y = act(W @ z + b)
        idx = 2
        for l in range(len(self.coupling_layers) - 1):
            W, b = theta_couple[idx], theta_couple[idx + 1]
            y = act(W @ y + b)
            idx += 2
        W, b = theta_couple[idx], theta_couple[idx + 1]
        out = W @ y + b  # (dims[i]+1,): L_i followed by m_i
        return out[:-1] @ xi + out[-1]

    def param_shapes(self):
        shapes = []
        self._own_slices = []
        for ni in self.dims:
            s = self._own_param_shapes(ni)
            self._own_slices.append((len(shapes), len(shapes) + len(s)))
            shapes += s
        self._couple_slices = []
        for i in range(self.N):
            s = self._coupling_param_shapes(i)
            self._couple_slices.append((len(shapes), len(shapes) + len(s)))
            shapes += s
        return shapes

    def costs(self, x, p, theta):
        if not self._own_slices:
            self.param_shapes()  # populate slice bookkeeping (shapes only, no-op on theta)
        J = []
        for i in range(self.N):
            o0, o1 = self._own_slices[i]
            c0, c1 = self._couple_slices[i]
            xi, x_mi = x[self.isi[i]], x[self.nisi[i]]
            hi = self._own_cost(xi, p, theta[o0:o1])
            ci = self._coupling_cost(xi, x_mi, p, theta[c0:c1])
            J.append(hi + ci)
        return J
