"""
Inverse learning of a monotone game-theoretic controller, as described in [1, Section 6.4].

Data are generated from a simulated CSTR (continuously stirred tank) with three inlet streams and
Torricelli (gravity-drain) outlet flow. The three inlet flows are held
constant at the steady-state values for a chosen operating point, and the
open-loop system is simulated from a nearby initial condition to show
convergence back to that operating point.

States:
    h   : liquid level in the tank             [m]
    C_A : concentration of A in the tank        [mol/L]
    C_B : concentration of B in the tank        [mol/L]

Open-loop model (see cstr_model.tex for the full derivation):

    A * dh/dt   = q1 + q2 + q3 - q_out(h)
    dC_A/dt     = ( q2*(C_A_in - C_A) - (q1+q3)*C_A ) / (A*h) - r
    dC_B/dt     = ( q3*(C_B_in - C_B) - (q1+q2)*C_B ) / (A*h) - r

with r = 0 unless Params.CHEMICAL_REACTIONS is True, in which case the
optional reaction A + B -> product is included with mass-action rate
r = k_reaction * C_A * C_B.

Outlet flow uses Torricelli's law:

    q_out(h) = c_v * sqrt(h)

[1] A. Bemporad, T. Tatarenko, "Learning Parametric Monotone Games,"   
    arXiv preprint 2609.02494, 2026, https://arxiv.org/abs/2609.02494. 

(C) 2026 A. Bemporad
"""

import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = plt.rcParams["font.size"] + 4
plt.ion()

# If True, collect a long random-setpoint closed-loop run and fit a 3-player 
# quadratic monotone game from it via inverse learning.
LEARN_GAME = True

# If True, simulate the closed loop with u(t) given by the Nash equilibrium
# of the learned game, solved every Ts=1s with a zero-order
# hold on the three flows (requires LEARN_GAME=True).
RUN_LEARNED_GAME = True

#LEARNING_METHOD = "inverse-SDP"
LEARNING_METHOD = "inverse-LS+SDP"
#LEARNING_METHOD = "inverse-NLS" # change rho in the call to inv.fit_nls() below to tune the regularization strength

Q_MAX_TOTAL = 0.5
Q_MIN_TOTAL = 0.4

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
class Params:
    A = 1.0           # tank cross-sectional area [m^2]
    c_v = 0.6         # Torricelli discharge coefficient, q_out = c_v*sqrt(h)
    C_A_in = 2.0      # feed concentration of A in stream q2 [mol/L]
    C_B_in = 1.5      # feed concentration of B in stream q3 [mol/L]
    h_min = 1e-6      # floor to avoid division by zero / sqrt of negative
    CHEMICAL_REACTIONS = True   # include A + B -> product reaction term
    k_reaction = 1.4            # mass-action rate constant [L/(mol*s)]


def outlet_flow(h, p: Params):
    """Torricelli's law: q_out = c_v * sqrt(h)."""
    return p.c_v * np.sqrt(max(h, 0.0))


# --------------------------------------------------------------------------
# Steady-state operating point for a desired (h0, CA0, CB0)
# --------------------------------------------------------------------------
def operating_point(h0, CA0, CB0, p: Params):
    """Solve the steady-state flows q1_0, q2_0, q3_0 that exactly hold the
    tank at (h0, CA0, CB0), including the reaction sink r0 = k*CA0*CB0 (when
    CHEMICAL_REACTIONS is True) in the full steady-state relations
        q_total_0 = c_v*sqrt(h0)
        q2_0*(C_A_in-CA0) - (q1_0+q3_0)*CA0 = A*h0*r0
        q3_0*(C_B_in-CB0) - (q1_0+q2_0)*CB0 = A*h0*r0
    Substituting q1_0 = q_total_0 - q2_0 - q3_0 decouples q2_0 and q3_0:
        q2_0 = (A*h0*r0 + CA0*q_total_0) / C_A_in
        q3_0 = (A*h0*r0 + CB0*q_total_0) / C_B_in
        q1_0 = q_total_0 - q2_0 - q3_0
    (r0 = 0 recovers the pure dilution/mixing balance.)
    """
    q_total_0 = p.c_v * np.sqrt(h0)
    r0 = p.k_reaction * CA0 * CB0 if p.CHEMICAL_REACTIONS else 0.0
    q2_0 = (p.A * h0 * r0 + CA0 * q_total_0) / p.C_A_in
    q3_0 = (p.A * h0 * r0 + CB0 * q_total_0) / p.C_B_in
    q1_0 = q_total_0 - q2_0 - q3_0
    if q1_0 < 0:
        raise ValueError(
            "Infeasible operating point: q1_0 < 0. Lower CA0/CB0 or "
            "raise h0 so the A+B feeds don't exceed the required total flow."
        )
    return q1_0, q2_0, q3_0, q_total_0


# --------------------------------------------------------------------------
# Open-loop dynamics: dx/dt = f(x, u), x = [h, C_A, C_B], u = [q1, q2, q3]
# --------------------------------------------------------------------------
def dynamics(x, u, p: Params):
    h, C_A, C_B = x
    q1, q2, q3 = u
    h_eff = max(h, p.h_min)

    r = p.k_reaction * C_A * C_B if p.CHEMICAL_REACTIONS else 0.0

    q_out = outlet_flow(h_eff, p)
    dh_dt = (q1 + q2 + q3 - q_out) / p.A
    dCA_dt = (q2 * (p.C_A_in - C_A) - (q1 + q3) * C_A) / (p.A * h_eff) - r
    dCB_dt = (q3 * (p.C_B_in - C_B) - (q1 + q2) * C_B) / (p.A * h_eff) - r

    return [dh_dt, dCA_dt, dCB_dt]


def make_open_loop_rhs(q1, q2, q3, p: Params):
    """Flows q1, q2, q3 held constant."""
    def rhs(t, x):
        return dynamics(x, (q1, q2, q3), p)

    return rhs


# --------------------------------------------------------------------------
# Linearization around a steady operating point: dx/dt = F*(x-x0) + G*(u-u0)
# --------------------------------------------------------------------------
def linearize(h0, CA0, CB0, p: Params = None):
    """Analytic Jacobians F = df/dx, G = df/du at the steady operating point
    (h0, CA0, CB0, q1_0, q2_0, q3_0) returned by operating_point().

    Since operating_point() solves for q1_0, q2_0, q3_0 that balance the
    reaction sink r0 = k*CA0*CB0 exactly, the concentration rows pick up a
    nonzero df/dh = -r0/h0 (zero only when CHEMICAL_REACTIONS is False). The
    level row still decouples from the concentrations (dh/dt never depends
    on C_A, C_B), and G does not depend on q1_0, q2_0, q3_0 or r0 at all.
    """
    p = p or Params()
    q1_0, q2_0, q3_0, q_tot0 = operating_point(h0, CA0, CB0, p)

    a_h = p.c_v / (2 * np.sqrt(h0))
    a_c = q_tot0 / (p.A * h0)
    k_r = p.k_reaction if p.CHEMICAL_REACTIONS else 0.0
    r0 = k_r * CA0 * CB0

    F = np.array([
        [-a_h / p.A,            0.0,               0.0],
        [-r0 / h0,   -a_c - k_r * CB0,       -k_r * CA0],
        [-r0 / h0,         -k_r * CB0,  -a_c - k_r * CA0],
    ])
    G = np.array([
        [1 / p.A,                    1 / p.A,               1 / p.A],
        [-CA0 / (p.A * h0), (p.C_A_in - CA0) / (p.A * h0), -CA0 / (p.A * h0)],
        [-CB0 / (p.A * h0), -CB0 / (p.A * h0), (p.C_B_in - CB0) / (p.A * h0)],
    ])
    return F, G, (q1_0, q2_0, q3_0)


# --------------------------------------------------------------------------
# Decentralized PI control, designed from the linearized model, with static
# decoupling of the remaining two flows treated as measured disturbances
# --------------------------------------------------------------------------
def design_pi_controllers(h0, CA0, CB0, p: Params = None, lambda_ratio=None):
    """Design one SISO PI loop per output (loop 1: q1->h, loop 2: q2->C_A,
    loop 3: q3->C_B), each tuned from its own diagonal first-order model
        d(delta_x_i)/dt = F_ii*delta_x_i + G_ii*delta_u_i
    via IMC/lambda tuning (Ti = tau_i, lambda_i = tau_i/lambda_ratio_i):
        tau_i = -1/F_ii,  K_i = -G_ii/F_ii = G_ii*tau_i
        Kp_i = tau_i/(K_i*lambda_i),  Ki_i = 1/(K_i*lambda_i)

    lambda_ratio is per-loop (dict keyed by "h","A","B"; a bare scalar
    applies to all three). The level loop (h) can be tuned more
    aggressively than the concentration loops: it decouples from C_A, C_B
    to first order (see linearize()), so it does not interact with the
    reaction-driven C_A/C_B coupling. With CHEMICAL_REACTIONS on, integral
    action on q2/q3 must keep growing to counteract ongoing A+B
    consumption, and (through the decoupling matrix Dm below) that steady
    pull leaks into q1; too fast an A/B tuning makes q1 saturate at its
    q1>=0 floor and the loop diverge. lambda_ratio["h"] is capped by the
    Ts=1s sample rate used elsewhere (simulate_learned_game_closed_loop):
    the closed-loop time constant tau_i/lambda_i must stay a few multiples
    of Ts, or a zero-order hold at Ts turns the fast continuous design into
    a lightly-damped/oscillatory sampled one -- lambda_ratio["h"] above ~5
    starts to alias this way even though the continuous design still looks
    stable on its own. The defaults keep the Ts=1s ZOH-sampled closed loop
    well damped over long simulation horizons.

    The other two flows q_j (j!=i), commanded by other loops and hence
    known, are *measured* disturbances for loop i: a static decoupling
    matrix, built from the off-diagonal G, converts the three independent
    PI outputs v = [v1,v2,v3] into actuator deltas delta = [dq1,dq2,dq3]
    such that G @ delta = diag(G) * v (the diagonal design's effect on
    dx/dt through G):

        Dm = inv(G) @ diag(diag(G))
        delta = Dm @ v

    which cancels, to first order, the cross-coupling q_j otherwise
    injects into output y_i.
    """
    p = p or Params()
    F, G, u0 = linearize(h0, CA0, CB0, p)

    if lambda_ratio is None:
        lambda_ratio = {"h": 4.0, "A": 0.2, "B": 0.1}
    elif np.isscalar(lambda_ratio):
        lambda_ratio = {name: lambda_ratio for name in ("h", "A", "B")}

    gains = {}
    for i, name in enumerate(("h", "A", "B")):
        tau_i = -1.0 / F[i, i]
        K_i = G[i, i] * tau_i
        lam_i = tau_i / lambda_ratio[name]
        Kp_i = tau_i / (K_i * lam_i)
        Ki_i = 1.0 / (K_i * lam_i)
        gains[name] = {"Kp": Kp_i, "Ki": Ki_i, "K": K_i, "tau": tau_i}

    Dm = np.linalg.inv(G) @ np.diag(np.diag(G))
    return gains, Dm, u0


# --------------------------------------------------------------------------
# Simulation driver
# --------------------------------------------------------------------------
def simulate_open_loop(t_span=(0, 150), n_points=1500,
                        h0=0.6, CA0=0.5, CB0=0.3,
                        x0=(0.3, 0.10, 0.05), params=None):
    p = params or Params()
    q1_0, q2_0, q3_0, _ = operating_point(h0, CA0, CB0, p)
    rhs = make_open_loop_rhs(q1_0, q2_0, q3_0, p)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45",
                     rtol=1e-8, atol=1e-10, max_step=0.5)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return {
        "t": sol.t, "h": sol.y[0], "C_A": sol.y[1], "C_B": sol.y[2],
        "h0": h0, "CA0": CA0, "CB0": CB0,
        "q1_0": q1_0, "q2_0": q2_0, "q3_0": q3_0,
    }


# --------------------------------------------------------------------------
# Step response: start at the steady state, then step the inlet flows
# --------------------------------------------------------------------------
def simulate_step_response(t_span=(0, 150), n_points=1500,
                            h0=0.6, CA0=0.5, CB0=0.3,
                            dq1=0.0, dq2=0.0, dq3=0.0,
                            t_step1=20.0, t_step2=20.0, t_step3=20.0,
                            params=None):
    """Start exactly at the steady state (h0, CA0, CB0) held by
    (q1_0, q2_0, q3_0), then apply a step change dqi to flow qi at its own
    time t_stepi (held constant thereafter) and observe the open-loop
    response. Steps at different times let the effect of each input be seen
    in isolation before the next one hits."""
    p = params or Params()
    q1_0, q2_0, q3_0, _ = operating_point(h0, CA0, CB0, p)

    def u_of_t(t):
        q1 = q1_0 + (dq1 if t >= t_step1 else 0.0)
        q2 = q2_0 + (dq2 if t >= t_step2 else 0.0)
        q3 = q3_0 + (dq3 if t >= t_step3 else 0.0)
        return (q1, q2, q3)

    def rhs(t, x):
        return dynamics(x, u_of_t(t), p)

    x0 = (h0, CA0, CB0)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45",
                     rtol=1e-8, atol=1e-10, max_step=0.5)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    q1s, q2s, q3s = zip(*(u_of_t(t) for t in sol.t))

    return {
        "t": sol.t, "h": sol.y[0], "C_A": sol.y[1], "C_B": sol.y[2],
        "q1": np.array(q1s), "q2": np.array(q2s), "q3": np.array(q3s),
        "h0": h0, "CA0": CA0, "CB0": CB0,
        "q1_0": q1_0, "q2_0": q2_0, "q3_0": q3_0,
        "t_step1": t_step1, "t_step2": t_step2, "t_step3": t_step3,
    }


def plot_step_response(res):
    fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)
    step_times = (res["t_step1"], res["t_step2"], res["t_step3"])

    axes[0].plot(res["t"], res["h"], color="tab:blue", label=r"$h$")
    axes[0].axhline(res["h0"], color="k", linestyle="--", label=r"$h_0$")
    axes[0].set_ylabel(r"level [m]")
    axes[0].set_title(r"Level")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(res["t"], res["C_A"], color="tab:orange", label=r"$C_A$")
    axes[1].axhline(res["CA0"], color="k", linestyle="--", label=r"$C_{A0}$")
    axes[1].set_ylabel(r"$C_A$ [mol/L]")
    axes[1].set_title(r"Concentration A")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(res["t"], res["C_B"], color="tab:green", label=r"$C_B$")
    axes[2].axhline(res["CB0"], color="k", linestyle="--", label=r"$C_{B0}$")
    axes[2].set_ylabel(r"$C_B$ [mol/L]")
    axes[2].set_title(r"Concentration B")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(res["t"], res["q1"], label=r"$q_1$ (water)")
    axes[3].plot(res["t"], res["q2"], label=r"$q_2$ ($A$ feed)")
    axes[3].plot(res["t"], res["q3"], label=r"$q_3$ ($B$ feed)")
    axes[3].set_ylabel(r"flow [m$^3$/s]")
    axes[3].set_xlabel(r"time [s]")
    axes[3].set_title(r"Inlet flows (steps at $t=$ " + str(step_times) + r")")
    axes[3].legend(); axes[3].grid(alpha=0.3)

    for ax in axes:
        for t_step in step_times:
            ax.axvline(t_step, color="gray", linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Closed-loop simulation: states = [h, C_A, C_B, Ih, IA, IB], decoupled PI
# --------------------------------------------------------------------------
def make_closed_loop_rhs(gains, Dm, u0, sp, p: Params):
    Kp_h, Ki_h = gains["h"]["Kp"], gains["h"]["Ki"]
    Kp_A, Ki_A = gains["A"]["Kp"], gains["A"]["Ki"]
    Kp_B, Ki_B = gains["B"]["Kp"], gains["B"]["Ki"]
    q1_0, q2_0, q3_0 = u0
    h_sp, CA_sp, CB_sp = sp

    def flows_at(t, x):
        """(q1, q2, q3) commanded by the three decoupled PI loops, saturated
        at the q_i >= 0 actuator limit."""
        h, C_A, C_B, Ih, IA, IB = x
        eh = h_sp(t) - h
        eA = CA_sp(t) - C_A
        eB = CB_sp(t) - C_B
        v = np.array([Kp_h * eh + Ki_h * Ih,
                      Kp_A * eA + Ki_A * IA,
                      Kp_B * eB + Ki_B * IB])
        delta = Dm @ v
        q = np.clip(np.array([q1_0, q2_0, q3_0]) + delta, 0.0, None)
        return q, (eh, eA, eB)

    def rhs(t, x):
        h, C_A, C_B, Ih, IA, IB = x
        (q1, q2, q3), (eh, eA, eB) = flows_at(t, x)

        dx = dynamics((h, C_A, C_B), (q1, q2, q3), p)

        return dx + [eh, eA, eB]

    return rhs, flows_at


def simulate_closed_loop(t_span=(0, 150), n_points=1500,
                          h0=0.6, CA0=0.5, CB0=0.3,
                          x0=None, h_sp=None, CA_sp=None, CB_sp=None,
                          lambda_ratio=None, params=None):
    """Closed-loop simulation with the three decentralized, decoupled PI
    loops from design_pi_controllers(), tracking (possibly time-varying)
    setpoints h_sp(t), CA_sp(t), CB_sp(t) (default: constant at h0/CA0/CB0),
    starting from x0 (default: the steady state itself)."""
    p = params or Params()
    gains, Dm, u0 = design_pi_controllers(h0, CA0, CB0, p, lambda_ratio)
    sp = (h_sp or (lambda t: h0), CA_sp or (lambda t: CA0), CB_sp or (lambda t: CB0))
    rhs, flows_at = make_closed_loop_rhs(gains, Dm, u0, sp, p)

    if x0 is None:
        x0 = (h0, CA0, CB0, 0.0, 0.0, 0.0)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45",
                     rtol=1e-8, atol=1e-10, max_step=0.5)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    q1s, q2s, q3s = [], [], []
    for t, xcol in zip(sol.t, sol.y.T):
        (q1, q2, q3), _ = flows_at(t, xcol)
        q1s.append(q1); q2s.append(q2); q3s.append(q3)

    return {
        "t": sol.t, "h": sol.y[0], "C_A": sol.y[1], "C_B": sol.y[2],
        "Ih": sol.y[3], "IA": sol.y[4], "IB": sol.y[5],
        "q1": np.array(q1s), "q2": np.array(q2s), "q3": np.array(q3s),
        "h_sp": np.array([sp[0](t) for t in sol.t]),
        "CA_sp": np.array([sp[1](t) for t in sol.t]),
        "CB_sp": np.array([sp[2](t) for t in sol.t]),
        "gains": gains, "Dm": Dm, "u0": u0,
    }


def plot_closed_loop(res, suptitle=None, q_bounds=None, q_total_ref=None):
    # States
    fig_states, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(res["t"], res["h"], color="tab:blue", label=r"$h$")
    axes[0].plot(res["t"], res["h_sp"], "k--")
    axes[0].set_ylabel(r"level [m]")
    axes[0].set_title(r"Level control (loop 1: $q_1$)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(res["t"], res["C_A"], color="tab:orange", label=r"$C_A$")
    axes[1].plot(res["t"], res["CA_sp"], "k--")
    axes[1].set_ylabel(r"$C_A$ [mol/L]")
    axes[1].set_title(r"Concentration A control (loop 2: $q_2$)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(res["t"], res["C_B"], color="tab:green", label=r"$C_B$")
    axes[2].plot(res["t"], res["CB_sp"], "k--")
    axes[2].set_ylabel(r"$C_B$ [mol/L]")
    axes[2].set_xlabel(r"time [s]")
    axes[2].set_title(r"Concentration B control (loop 3: $q_3$)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig_states.tight_layout()
    if suptitle:
        fig_states.suptitle(suptitle)

    # Inputs
    fig_inputs, axes_inputs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes_inputs[0].plot(res["t"], res["q1"], label=r"$q_1$ (water)")
    axes_inputs[0].plot(res["t"], res["q2"], label=r"$q_2$ ($A$ feed)")
    axes_inputs[0].plot(res["t"], res["q3"], label=r"$q_3$ ($B$ feed)")
    axes_inputs[0].set_ylabel(r"flow [m$^3$/s]")
    axes_inputs[0].set_title(r"Manipulated flows")
    axes_inputs[0].legend()
    axes_inputs[0].grid(alpha=0.3)

    q_total = res["q1"] + res["q2"] + res["q3"]
    axes_inputs[1].plot(res["t"], q_total, color="tab:purple", label=r"$q_1+q_2+q_3 (GNE)$")
    if q_total_ref is not None:
        axes_inputs[1].plot(res["t"], q_total_ref, color="k", linestyle=":",
                             label=r"$q_1+q_2+q_3$ (PI)")
    if q_bounds is not None:
        q_min_total, q_max_total = q_bounds
        axes_inputs[1].axhline(q_max_total, color="gray", linestyle="--")
        axes_inputs[1].axhline(q_min_total, color="gray", linestyle="--")
    axes_inputs[1].set_ylabel(r"flow [m$^3$/s]")
    axes_inputs[1].set_xlabel(r"time [s]")
    axes_inputs[1].set_title(r"Total inlet flow ($q_1+q_2+q_3$)")
    if q_total_ref is not None:
        axes_inputs[1].legend()
    axes_inputs[1].grid(alpha=0.3)

    fig_inputs.tight_layout()
    if suptitle:
        fig_inputs.suptitle(suptitle + " - inputs")

    plt.show()


# --------------------------------------------------------------------------
# Random-setpoint data collection (used by LEARN_GAME to build the
# best-response training set)
# --------------------------------------------------------------------------
def random_piecewise_signal(t_span, base, amplitude, seg_time_range, rng):
    """Piecewise-constant random signal around `base` +/- `amplitude`, with
    segment durations drawn uniformly from seg_time_range = (t_min, t_max)."""
    t0, t1 = t_span
    breakpoints = [t0]
    while breakpoints[-1] < t1:
        breakpoints.append(breakpoints[-1] + rng.uniform(*seg_time_range))
    breakpoints = np.array(breakpoints)
    values = base + rng.uniform(-amplitude, amplitude, size=len(breakpoints))

    def signal(t):
        idx = np.searchsorted(breakpoints, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return signal


def simulate_random_setpoint_run(Tsim=1999.0, Ts=1.0,
                                  h0=0.6, CA0=0.5, CB0=0.3,
                                  seg_time_range=(60.0, 150.0),
                                  h_amp=0.1, CA_amp=0.1, CB_amp=0.05,
                                  seed=None, params=None):
    """Long closed-loop run with random piecewise-constant setpoints for
    h, C_A, C_B, used to collect a training dataset (e.g. for learning a
    surrogate model of the coupled dynamics). The state is sampled every Ts
    seconds over a horizon of Tsim seconds."""
    rng = np.random.default_rng(seed)
    h_sp = random_piecewise_signal((0.0, Tsim), h0, h_amp, seg_time_range, rng)
    CA_sp = random_piecewise_signal((0.0, Tsim), CA0, CA_amp, seg_time_range, rng)
    CB_sp = random_piecewise_signal((0.0, Tsim), CB0, CB_amp, seg_time_range, rng)

    n_points = int(round(Tsim / Ts)) + 1
    res = simulate_closed_loop(t_span=(0.0, Tsim), n_points=n_points,
                                h0=h0, CA0=CA0, CB0=CB0,
                                h_sp=h_sp, CA_sp=CA_sp, CB_sp=CB_sp,
                                params=params)
    # initial condition and setpoint callables, so this scenario can be
    # replayed exactly (e.g. by simulate_learned_game_closed_loop)
    res["h0"], res["CA0"], res["CB0"] = h0, CA0, CB0
    res["h_sp_fn"], res["CA_sp_fn"], res["CB_sp_fn"] = h_sp, CA_sp, CB_sp
    return res


# --------------------------------------------------------------------------
# Inverse learning of a 3-player quadratic monotone game (LEARN_GAME) and
# closed-loop test of the learned game (RUN_LEARNED_GAME)
#
# Decision variables: x1, x2, x3 = q1, q2, q3, the three inlet flows
# commanded at each sample time t=k*Ts (agent 1 <-> level loop, agent 2 <->
# C_A loop, agent 3 <-> C_B loop, mirroring the three decentralized PI
# loops of design_pi_controllers()).
# Parameter: p(t) = [h, C_A, C_B, h_sp, CA_sp, CB_sp, Ih, IA, IB](t), i.e.
# the current states, current references, and PI error-integrator states.
# --------------------------------------------------------------------------
def _ensure_monotone_games_importable():
    """monotone_games lives alongside this file's 'python' directory, but
    may not be on the default path if this script is run from elsewhere;
    add it to sys.path the first time it is needed."""
    project_python_dir = os.path.dirname(os.path.abspath(__file__))
    if project_python_dir not in sys.path:
        sys.path.insert(0, project_python_dir)


def build_br_dataset(res):
    """Best-response triples (Xbr, P, agent_idx) for inverse-learning the
    quadratic game from one closed-loop trajectory: every sample time
    contributes one row per agent, since all three flows are observed
    simultaneously (Xbr = [q1,q2,q3](t) for every row, only agent_idx
    differs)."""
    X = np.column_stack([res["q1"], res["q2"], res["q3"]])
    P = np.column_stack([res["h"], res["C_A"], res["C_B"],
                          res["h_sp"], res["CA_sp"], res["CB_sp"],
                          res["Ih"], res["IA"], res["IB"]])
    T = X.shape[0]
    Xbr = np.tile(X, (3, 1))
    Pbr = np.tile(P, (3, 1))
    agent_idx = np.repeat(np.arange(3), T)
    return Xbr, Pbr, agent_idx


def learned_best_response(cost_model, theta, x, p, i):
    """Closed-form best response of a StructuredMonotoneCost(potential=None)
    model: argmin_{x_i} J_i(x,p;theta) = -A_ii^{-1}(A_{i,-i} x_{-i} + q_i)."""
    import jax.numpy as jnp
    theta_quad = theta[:cost_model.n_param_quad]
    A_pg, q = cost_model.pseudogradient_matrix(jnp.asarray(p), theta_quad)
    A_pg, q = np.asarray(A_pg), np.asarray(q)
    isi, nisi = cost_model.isi[i], cost_model.nisi[i]
    return -np.linalg.solve(A_pg[np.ix_(isi, isi)], A_pg[np.ix_(isi, nisi)] @ x[nisi] + q[isi])


def fit_quadratic_game(res, mu=0., train_frac=0.8, val_frac=0.1, seed=0, solver="SCS",
                        method=None, rho_nls=1e-12, adam_epochs=1000, lbfgs_epochs=5000,
                        n_seeds=None):
    """Fit the 3-player quadratic monotone game from a closed-loop
    trajectory. `method` (defaults to the module-level LEARNING_METHOD flag)
    selects QuadraticInverseSolver's fitting approach, following
    example_quad_game.py's FIT_INVERSE_LEARNING branch:
        "inverse-SDP"    : fit_sdp, a single convex SDP (global optimum;
                           requires cvxpy), constant-in-p A.
        "inverse-LS+SDP" : fit_ls_sdp_cascade, an unconstrained per-agent
                           OLS fit followed by a small SDP reconciling
                           A,q0,q1 (requires cvxpy), constant-in-p A.
        "inverse-NLS"    : fit_nls, nonlinear least squares via jax_sysid
                           (Adam + L-BFGS) from n_seeds parallel initial
                           guesses -- one warm-started from the LS+SDP
                           cascade solution, the rest random -- keeping the
                           run with the lowest held-out (validation) best-
                           response residual; allows A to depend affinely
                           on p too.
    Returns (cost_model, theta, diagnostics), diagnostics reporting the
    held-out (test-split) best-response error."""
    method = method or LEARNING_METHOD
    _ensure_monotone_games_importable()
    from monotone_games import QuadraticInverseSolver

    Xbr_all, P_all, idx_all = build_br_dataset(res)
    npar = P_all.shape[1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(Xbr_all.shape[0])
    n_total = perm.shape[0]
    n1 = int(round(train_frac * n_total))
    n2 = n1 + int(round(val_frac * n_total))
    train_idx, val_idx, test_idx = perm[:n1], perm[n1:n2], perm[n2:]
    Xbr, P, agent_idx = Xbr_all[train_idx], P_all[train_idx], idx_all[train_idx]
    Xbr_val, P_val, idx_val = Xbr_all[val_idx], P_all[val_idx], idx_all[val_idx]
    Xbr_test, P_test, idx_test = Xbr_all[test_idx], P_all[test_idx], idx_all[test_idx]

    inv = QuadraticInverseSolver([1, 1, 1], npar, mu=mu)

    if method == "inverse-SDP":
        t0 = time.time()
        fit_result = inv.fit_sdp(Xbr, P, agent_idx, solver=solver)
        cost_model, theta = inv.to_cost_model(fit_result)
        training_time = time.time() - t0
    elif method == "inverse-LS+SDP":
        t0 = time.time()
        fit_result = inv.fit_ls_sdp_cascade(Xbr, P, agent_idx, solver=solver)
        cost_model, theta = inv.to_cost_model(fit_result)
        training_time = time.time() - t0
    elif method == "inverse-NLS":
        from joblib import cpu_count
        t0 = time.time()
        res_cascade = inv.fit_ls_sdp_cascade(Xbr, P, agent_idx, solver=solver)
        res_nls = inv.fit_nls(
            Xbr, P, agent_idx, seeds=np.arange(seed, seed + (n_seeds or cpu_count())),
            warm_start=res_cascade, val_data=(Xbr_val, P_val, idx_val),
            rho=1.e-6, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)
        cost_model, theta = inv.to_cost_model(res_nls)
        training_time = time.time() - t0
    else:
        raise ValueError(f"Unknown LEARNING_METHOD {method!r}; expected "
                          "'inverse-SDP', 'inverse-LS+SDP', or 'inverse-NLS'.")

    br_pred = np.array([learned_best_response(cost_model, theta, x, p, i)
                         for x, p, i in zip(Xbr_test, P_test, idx_test)]).reshape(-1)
    br_true = np.array([x[cost_model.isi[i]] for x, i in zip(Xbr_test, idx_test)]).reshape(-1)
    br_error = np.abs(br_pred - br_true)

    diagnostics = {
        "name": method, "cpu_time": training_time,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "br_error_mean": float(br_error.mean()), "br_error_max": float(br_error.max()),
    }
    return cost_model, theta, diagnostics


AGENT_NAMES = ("h", "A", "B")   # agent i controls q_{i+1}, tracking loop i
PARAM_NAMES = ("h", "C_A", "C_B", "h_sp", "CA_sp", "CB_sp", "Ih", "IA", "IB")


def print_learned_costs(cost_model, theta, agent_names=AGENT_NAMES, param_names=PARAM_NAMES):
    """Print each agent's learned quadratic cost
        J_i(x,p) = 0.5*A_ii*x_i^2 + x_i*sum_{j!=i} A_ij*x_j + q_i(p)*x_i,
        q_i(p) = q1_i.p + q0_i (affine in p),
    reading A and q directly off theta (phi_i == 0 identically: best-
    response data carries no information about the x_i-independent terms,
    see QuadraticInverseSolver.to_cost_model). A is evaluated at p=0, but is
    constant-A by construction (cascade/SDP fit) so this equals its value
    at every other p; only q depends on p."""
    theta_quad = theta[:cost_model.n_param_quad]
    npar = cost_model.npar
    A, q0 = cost_model.pseudogradient_matrix(np.zeros(npar), theta_quad)
    A = np.asarray(A)
    q1 = np.asarray(theta_quad[4])   # (nx, npar): p-affine coefficients of q
    q0 = np.asarray(q0)              # (nx,): constant term of q (at p=0)

    print("\nLearned quadratic costs "
          "J_i(x,p) = 0.5*A_ii*x_i^2 + x_i*sum_{j!=i} A_ij*x_j + q_i(p)*x_i, "
          "q_i(p) = q1_i.p + q0_i:")
    for i, name in enumerate(agent_names):
        others = [j for j in range(cost_model.N) if j != i]
        coupling = " ".join(f"{A[i, j]:+.4f}*x_{agent_names[j]}" for j in others)
        q_terms = " ".join(f"{q1[i, k]:+.4f}*{pname}" for k, pname in enumerate(param_names))
        print(f"  Agent {name} (decision x_{name} = q{i + 1}):")
        print(f"    J_{name} = 0.5*{A[i, i]:.4f}*x_{name}^2 + x_{name}*( {coupling} )")
        print(f"             + ( {q_terms} {q0[i]:+.4f} )*x_{name}")
    return A, q1, q0


def _latex_sci(v, bold=False):
    """Format v as LaTeX scientific notation, e.g. $1.23\\times10^{-4}$
    (matches example_quad_game.py's helper of the same name)."""
    if v is None or np.isnan(v):
        return "--"
    mantissa, exponent = f"{v:.2e}".split("e")
    body = f"{mantissa} \\times 10^{{{int(exponent)}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


def make_latex_table(rows):
    """LaTeX table of the learned quadratic game's fit(s): training time and
    held-out best-response error, one row per inverse-learning method
    (mirrors example_quad_game.py's make_latex_table, minus the NE-error
    column that script's synthetic-game setup can evaluate but this
    closed-loop trajectory cannot)."""
    method_names = {
        "inverse-SDP": "SDP",
        "inverse-LS+SDP": "LS+SDP",
        "inverse-NLS": "NLS",
    }
    header = r"  method & time (s) & BR error \\"
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \renewcommand{\arraystretch}{1.} % Adjust row separation",
        r"  \caption{Quadratic monotone game learned from the CSTR closed-loop trajectory.}",
        r"  \label{tab:cstr_game}",
        r"  \begin{tabular}{l|rr}",
        r"  \hline",
        header,
        r"  \hline",
    ]
    best = {
        "cpu_time": min(r["cpu_time"] for r in rows),
        "br_error_mean": min(r["br_error_mean"] for r in rows),
    }
    for r in rows:
        cells = [method_names.get(r["name"], r["name"])]
        s = f"{r['cpu_time']:.4f}"
        cells.append(r"\textbf{" + s + "}"
                     if r["cpu_time"] == best["cpu_time"] else s)
        cells.append(_latex_sci(r["br_error_mean"], bold=(r["br_error_mean"] == best["br_error_mean"])))
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def simulate_learned_game_closed_loop(cost_model, theta, t_span=(0, 150), Ts=1.0,
                                       h0=0.6, CA0=0.5, CB0=0.3,
                                       x0=None, h_sp=None, CA_sp=None, CB_sp=None,
                                       q_max_total=Q_MAX_TOTAL, q_min_total=Q_MIN_TOTAL, params=None):
    """Closed-loop simulation where u(t) = [q1,q2,q3] is the Nash equilibrium
    of the learned quadratic game at p(t) = [state(t), setpoints(t), integral
    states(t)], solved every Ts seconds and held constant (zero-order hold)
    until the next sample, subject to q_i >= 0 and q_min_total <= q1+q2+q3
    <= q_max_total. The learned game carries no constraint information, so
    these are enforced via EquilibriumSolver.solve_quadratic, which
    delegates to nashopt.lq.qp_gnep's KKT-QP solver (daqp backend, requires
    nashopt>=1.3.0) whenever constraints are set. Integral states Ih, IA, IB
    evolve continuously between samples; anti-windup freezes loop i's
    integrator whenever the constrained q_i differs from the unconstrained
    equilibrium (same clamping rule as make_closed_loop_rhs), so p(t) stays
    consistent with the trajectory the game was learned from."""
    _ensure_monotone_games_importable()
    from monotone_games import EquilibriumSolver

    p = params or Params()
    eq_solver_unc = EquilibriumSolver(cost_model)
    eq_solver = EquilibriumSolver(cost_model, constraints={
        "lb": np.zeros(3),
        "A": np.array([[1.0, 1.0, 1.0],
                        [-1.0, -1.0, -1.0]]),
        "b": np.array([q_max_total, -q_min_total]),
    })
    sp = (h_sp or (lambda t: h0), CA_sp or (lambda t: CA0), CB_sp or (lambda t: CB0))

    x = np.array([h0, CA0, CB0, 0.0, 0.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)

    n_steps = int(round((t_span[1] - t_span[0]) / Ts))
    t_hist = t_span[0] + np.arange(n_steps + 1) * Ts

    x_hist = np.zeros((6, n_steps + 1))
    q_hist = np.zeros((3, n_steps + 1))
    x_hist[:, 0] = x

    for k in range(n_steps):
        t = t_hist[k]
        h, C_A, C_B, Ih, IA, IB = x
        p_t = np.array([h, C_A, C_B, sp[0](t), sp[1](t), sp[2](t), Ih, IA, IB])
        q_unsat = np.asarray(eq_solver_unc.solve_quadratic(p_t, theta).x)
        q = np.asarray(eq_solver.solve_quadratic(p_t, theta).x)
        not_sat = np.isclose(q, q_unsat)
        q_hist[:, k] = q

        def rhs(tt, xx, q=q, not_sat=not_sat):
            hh, CAv, CBv, Ihh, IAv, IBv = xx
            dx = dynamics((hh, CAv, CBv), tuple(q), p)
            e = (sp[0](tt) - hh, sp[1](tt) - CAv, sp[2](tt) - CBv)
            dI = [e[j] if not_sat[j] else 0.0 for j in range(3)]
            return dx + dI

        sol_ivp = solve_ivp(rhs, (t, t + Ts), x, t_eval=[t + Ts], method="RK45",
                             rtol=1e-8, atol=1e-10, max_step=Ts / 10)
        if not sol_ivp.success:
            raise RuntimeError(f"Integration failed: {sol_ivp.message}")
        x = sol_ivp.y[:, -1]
        x_hist[:, k + 1] = x

    q_hist[:, -1] = q_hist[:, -2]

    return {
        "t": t_hist, "h": x_hist[0], "C_A": x_hist[1], "C_B": x_hist[2],
        "Ih": x_hist[3], "IA": x_hist[4], "IB": x_hist[5],
        "q1": q_hist[0], "q2": q_hist[1], "q3": q_hist[2],
        "h_sp": np.array([sp[0](t) for t in t_hist]),
        "CA_sp": np.array([sp[1](t) for t in t_hist]),
        "CB_sp": np.array([sp[2](t) for t in t_hist]),
    }


def plot_results(res):
    fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

    axes[0].plot(res["t"], res["h"], color="tab:blue", label=r"$h$")
    axes[0].axhline(res["h0"], color="k", linestyle="--", label=r"$h_0$")
    axes[0].set_ylabel(r"level [m]")
    axes[0].set_title(r"Level")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(res["t"], res["C_A"], color="tab:orange", label=r"$C_A$")
    axes[1].axhline(res["CA0"], color="k", linestyle="--", label=r"$C_{A0}$")
    axes[1].set_ylabel(r"$C_A$ [mol/L]")
    axes[1].set_title(r"Concentration A")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(res["t"], res["C_B"], color="tab:green", label=r"$C_B$")
    axes[2].axhline(res["CB0"], color="k", linestyle="--", label=r"$C_{B0}$")
    axes[2].set_ylabel(r"$C_B$ [mol/L]")
    axes[2].set_xlabel(r"time [s]")
    axes[2].set_title(r"Concentration B")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


learned_cost_model = learned_theta = None
if LEARN_GAME:
    print("Collecting a long random-setpoint run to learn a quadratic "
            "monotone game (agents = q1,q2,q3; p = state, setpoints, "
            "integral states)...")
    game_data = simulate_random_setpoint_run(Tsim=1999.0, Ts=1.0,
                                              seg_time_range=(50.0, 50.0), seed=1)
    learned_cost_model, learned_theta, diag = fit_quadratic_game(game_data)
    print(f"Learned quadratic game from {diag['n_train']} training / "
            f"{diag['n_test']} held-out best-response samples: "
            f"held-out BR error mean={diag['br_error_mean']:.4e}  "
            f"max={diag['br_error_max']:.4e}")

    print("\n\n% ===== LaTeX Table (fit / BR-error results) =====\n")
    print(make_latex_table([diag]))

    A, q1, q0 = print_learned_costs(learned_cost_model, learned_theta)

    # Plot the random-setpoint trajectory used to generate the training data
    plot_closed_loop(
        game_data,
        suptitle=r"Training data: random-setpoint closed-loop run",
    )

if RUN_LEARNED_GAME:
    if learned_cost_model is None:
        raise ValueError("RUN_LEARNED_GAME requires LEARN_GAME=True "
                            "to fit the game first.")

    h0, CA0, CB0 = 0.6, 0.5, 0.3

    def h_sp(t):
        return h0 if t < 80 else h0 + 0.2

    def CA_sp(t):
        return CA0 if t < 40 else CA0 + 0.2

    def CB_sp(t):
        return CB0 if t < 120 else CB0 - 0.1

    learned_res = simulate_learned_game_closed_loop(
        learned_cost_model, learned_theta, t_span=(0, 200), Ts=1.0,
        h0=h0, CA0=CA0, CB0=CB0, h_sp=h_sp, CA_sp=CA_sp, CB_sp=CB_sp)
    plot_closed_loop(learned_res, suptitle=r"Learned-game NE, solved every $T_s=1$s (ZOH)",
                      q_bounds=(Q_MIN_TOTAL, Q_MAX_TOTAL))

    print("\nLearned-game NE closed-loop tracking (setpoint steps at t=40/80/120):")
    print(" t [s]     h       C_A      C_B")
    n = len(learned_res["t"])
    for i in range(0, n, max(1, n // 12)):
        print(f"{learned_res['t'][i]:6.1f}  {learned_res['h'][i]:6.3f}  "
                f"{learned_res['C_A'][i]:7.4f}  {learned_res['C_B'][i]:7.4f}")

    # Replay the learned game (ZOH + constraints) on the test dataset's own
    # initial condition/setpoints, and compare against the true (continuous,
    # unconstrained) PI trajectory that generated it.
    test_res = simulate_learned_game_closed_loop(
        learned_cost_model, learned_theta, t_span=(0, game_data["t"][-1]), Ts=1.0,
        h0=game_data["h0"], CA0=game_data["CA0"], CB0=game_data["CB0"],
        h_sp=game_data["h_sp_fn"], CA_sp=game_data["CA_sp_fn"], CB_sp=game_data["CB_sp_fn"])
    plot_closed_loop(test_res, suptitle=r"Learned-game NE vs. test-dataset setpoints (ZOH)",
                      q_bounds=(Q_MIN_TOTAL, Q_MAX_TOTAL),
                      q_total_ref=game_data["q1"] + game_data["q2"] + game_data["q3"])

    h_err = np.abs(test_res["h"] - game_data["h"])
    ca_err = np.abs(test_res["C_A"] - game_data["C_A"])
    cb_err = np.abs(test_res["C_B"] - game_data["C_B"])
    print("\nLearned-game NE vs. true PI trajectory, replayed on the test "
          "dataset's own initial condition/setpoints:")
    print(f"  |h error|   : mean={h_err.mean():.4e}  max={h_err.max():.4e}")
    print(f"  |C_A error| : mean={ca_err.mean():.4e}  max={ca_err.max():.4e}")
    print(f"  |C_B error| : mean={cb_err.mean():.4e}  max={cb_err.max():.4e}")

print("Eigenvalues of symmetric part of pseudogradient matrix (A+A')/2: ", np.linalg.eigvals((A+A.T)/2))

if LEARN_GAME or RUN_LEARNED_GAME:
    raise SystemExit

results = simulate_open_loop()

print("Steady-state operating point:",
        {"h0": results["h0"], "CA0": results["CA0"], "CB0": results["CB0"]})
print("Steady-state flows:",
        {k: round(results[k], 4) for k in ("q1_0", "q2_0", "q3_0")})
if Params.CHEMICAL_REACTIONS:
    print("Note: CHEMICAL_REACTIONS=True, so q1_0/q2_0/q3_0 already balance "
            "the reaction sink k*CA0*CB0, making (h0, CA0, CB0) an exact "
            "steady state.")

F, G, u0 = linearize(results["h0"], results["CA0"], results["CB0"])
print("\nLinearized model dx/dt = F*(x-x0) + G*(u-u0), x=[h,C_A,C_B], u=[q1,q2,q3]")
print("F =\n", F)
print("G =\n", G)
print("eigenvalues(F) =", np.linalg.eigvals(F))

plot_results(results)

print("\n t [s]     h       C_A      C_B")
for i in range(0, len(results["t"]), len(results["t"]) // 12):
    print(f"{results['t'][i]:6.1f}  {results['h'][i]:6.3f}  "
            f"{results['C_A'][i]:7.4f}  {results['C_B'][i]:7.4f}")

# Step response: start at the steady state, then step q2 at t=20,
# q1 at t=60, and q3 at t=100, each observed before the next one hits
step_res = simulate_step_response(
    t_span=(0, 250), n_points=2500,
    h0=results["h0"], CA0=results["CA0"], CB0=results["CB0"],
    dq1=0.1, dq2=0.2, dq3=0.15,
    t_step1=60.0, t_step2=20.0, t_step3=100.0)
plot_step_response(step_res)

print("\nStep response (dq2=+0.2 @t=20, dq1=+0.1 @t=60, dq3=+0.15 @t=100):")
print(" t [s]     h       C_A      C_B")
for i in range(0, len(step_res["t"]), len(step_res["t"]) // 12):
    print(f"{step_res['t'][i]:6.1f}  {step_res['h'][i]:6.3f}  "
            f"{step_res['C_A'][i]:7.4f}  {step_res['C_B'][i]:7.4f}")

# Decentralized PI control, decoupled via measured disturbances (q_j)
h0, CA0, CB0 = results["h0"], results["CA0"], results["CB0"]
gains, Dm, u0 = design_pi_controllers(h0, CA0, CB0)
print("\nDecoupled PI gains (from IMC/lambda tuning on the linearized model):")
for loop, g in gains.items():
    print(f"  loop {loop}: Kp={g['Kp']:.4f}  Ki={g['Ki']:.4f}  "
            f"(K={g['K']:.4f}, tau={g['tau']:.4f})")
print("Decoupling matrix Dm =\n", Dm)

def h_sp(t):
    return h0 if t < 80 else h0 + 0.2

def CA_sp(t):
    return CA0 if t < 40 else CA0 + 0.2

def CB_sp(t):
    return CB0 if t < 120 else CB0 - 0.1

closed_res = simulate_closed_loop(t_span=(0, 200), n_points=2000,
                                    h0=h0, CA0=CA0, CB0=CB0,
                                    h_sp=h_sp, CA_sp=CA_sp, CB_sp=CB_sp)
plot_closed_loop(closed_res)

print("\nClosed-loop tracking (setpoint steps at t=40/80/120):")
print(" t [s]     h       C_A      C_B")
for i in range(0, len(closed_res["t"]), len(closed_res["t"]) // 12):
    print(f"{closed_res['t'][i]:6.1f}  {closed_res['h'][i]:6.3f}  "
            f"{closed_res['C_A'][i]:7.4f}  {closed_res['C_B'][i]:7.4f}")
