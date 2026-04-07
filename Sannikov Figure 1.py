import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import solve_ivp
from matplotlib.ticker import ScalarFormatter

# =========================================================
# Sannikov (2007), Figure 1
# u(c)=sqrt(c), h(a)=0.5 a^2 + 0.4 a, r=0.1, sigma=1
# =========================================================

r = 0.1
sigma = 1.0
A_MAX = 12.0          # numerical truncation only
WMAX = 12.0             # numerical truncation only
MIN_WGP = 0.05       # reject degenerate solutions that hit F0 almost immediately

DEBUG_SHOOT = True
DEBUG_ODE = False
DEBUG_POLICY = False

# ---------------------------------------------------------
# primitives given by Figure 1
# ---------------------------------------------------------
def u(c):
    c = np.maximum(c, 0.0)
    return np.sqrt(c)

def h(a):
    a = np.maximum(a, 0.0)
    return 0.5 * a**2 + 0.4 * a

def gamma(a):
    # paper: gamma(a)=h'(a) when A is an interval and h is differentiable
    a = np.maximum(a, 0.0)
    return a + 0.4

def F0(W):
    # F0(u(c)) = -c, u(c)=sqrt(c) => c=W^2
    return -W**2

def F0p(W):
    return -2.0 * W

# ---------------------------------------------------------
# consumption from equation (11)
# maximize -c - p*u(c), where p = F'(W)
#
# for u(c)=sqrt(c):
# if p >= 0, c*=0
# if p < 0, c*=p^2/4
# ---------------------------------------------------------
def c_star_from_p(p):
    if p >= 0:
        return 0.0
    return 0.25 * p**2

# ---------------------------------------------------------
# equation (6): q = min_a RHS(a)
# q(W)=F''(W)
# ---------------------------------------------------------
def q_rhs_given_a(a, W, F, p):
    a = min(max(a, 0.0), A_MAX)
    c = c_star_from_p(p)

    denom = 0.5 * r * sigma**2 * gamma(a)**2
    numer = F - a + c - p * (W - u(c) + h(a))
    return numer / denom

def q_and_a_star(W, F, p):
    obj = lambda a: q_rhs_given_a(a, W, F, p)

    res = minimize_scalar(
        obj,
        bounds=(0.0, A_MAX),
        method='bounded',
        options={'xatol': 1e-10}
    )

    a_star = float(res.x)
    a_star = min(max(a_star, 0.0), A_MAX)
    q_star = float(q_rhs_given_a(a_star, W, F, p))

    if DEBUG_POLICY:
        c = c_star_from_p(p)
        print(
            f"[policy] W={W:.6f}, F={F:.6f}, p={p:.6f}, "
            f"a*={a_star:.6f}, c*={c:.6f}, q={q_star:.6f}"
        )

    return q_star, a_star

# ---------------------------------------------------------
# ODE system
# y = [F, p], p = F'
# y' = [p, q]
# ---------------------------------------------------------
_last_bucket = -1

def ode_system(W, y):
    global _last_bucket
    F, p = y
    q, a = q_and_a_star(W, F, p)
    c = c_star_from_p(p)
    drift = r * (W - u(c) + h(a))

    if DEBUG_ODE:
        bucket = int(W * 20)
        if bucket > _last_bucket:
            _last_bucket = bucket
            print(
                f"[ode] W={W:.6f}, F={F:.6f}, p={p:.6f}, q={q:.6f}, "
                f"a={a:.6f}, c={c:.6f}, drift={drift:.6f}, F-F0={F-F0(W):.6f}"
            )

    return [p, q]

# ---------------------------------------------------------
# event: hit retirement frontier F=F0
# ---------------------------------------------------------
def hit_retirement_event(W, y):
    F, p = y
    return F - F0(W)

hit_retirement_event.terminal = True
hit_retirement_event.direction = -1

def solve_path_given_p0(p0, Wmax_init=WMAX, Wmax_cap=40.0):
    Wmax = Wmax_init

    while True:
        sol = solve_ivp(
            ode_system,
            (0.0, Wmax),
            [0.0, p0],
            events=hit_retirement_event,
            dense_output=False,
            max_step=0.002,
            rtol=1e-8,
            atol=1e-10
        )

        if sol.t_events[0].size > 0:
            return sol, Wmax, True

        if Wmax >= Wmax_cap:
            return sol, Wmax, False

        Wmax *= 2.0

# ---------------------------------------------------------
# shooting residual
# ---------------------------------------------------------
def shooting_residual(p0, verbose=False):
    sol, Wmax_used, did_hit = solve_path_given_p0(p0)

    if not did_hit:
        if verbose:
            F_end = sol.y[0, -1]
            W_end = sol.t[-1]
            print(
                f"[shoot] p0={p0:.8f}: no hit up to W={W_end:.4f}, "
                f"F-F0 at end = {F_end - F0(W_end):.6f} -> treat as overshoot"
            )
        # 关键：把 no-hit 当作“根右边”的信号
        return +1.0

    Wgp = float(sol.t_events[0][0])
    Fgp, pgp = sol.y_events[0][0]
    gap = pgp - F0p(Wgp)

    if verbose:
        print(
            f"[shoot] p0={p0:.8f}, Wgp={Wgp:.6f}, "
            f"Fgp={Fgp:.6f}, F0(Wgp)={F0(Wgp):.6f}, "
            f"pgp={pgp:.6f}, F0p(Wgp)={F0p(Wgp):.6f}, "
            f"smooth_gap={gap:.6e}"
        )

    return gap

# ---------------------------------------------------------
# search for p0，在最优区间寻找确切最优解
# ---------------------------------------------------------
def find_optimal_p0():
    grid = np.linspace(0.5, 3.2, 180)

    prev_p0 = None
    prev_mode = None   # "hit" or "overshoot"
    prev_gap = None

    print("=== shooting scan begins ===")

    bracket = None

    for p0 in grid:
        sol, _, did_hit = solve_path_given_p0(p0)

        if did_hit:
            Wgp = float(sol.t_events[0][0])
            Fgp, pgp = sol.y_events[0][0]
            gap = pgp - F0p(Wgp)

            print(
                f"[shoot] p0={p0:.8f}, Wgp={Wgp:.6f}, "
                f"Fgp={Fgp:.6f}, F0(Wgp)={F0(Wgp):.6f}, "
                f"pgp={pgp:.6f}, F0p(Wgp)={F0p(Wgp):.6f}, "
                f"smooth_gap={gap:.6e}"
            )

            mode = "hit"
        else:
            W_end = sol.t[-1]
            F_end = sol.y[0, -1]
            print(
                f"[shoot] p0={p0:.8f}: no hit up to W={W_end:.4f}, "
                f"F-F0 at end = {F_end - F0(W_end):.6f} -> treat as overshoot"
            )
            gap = None
            mode = "overshoot"

        if prev_p0 is not None:
            # want largest slope: left point still hits, right point overshoots
            if prev_mode == "hit" and mode == "overshoot":
                bracket = (prev_p0, p0)
                print(f"=== found bracket: {bracket} ===")
                break

        prev_p0, prev_mode, prev_gap = p0, mode, gap

    if bracket is None:
        raise RuntimeError("No hit/overshoot bracket found.")

    lo, hi = bracket

    # bisection on the boundary between hit and overshoot
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        sol, _, did_hit = solve_path_given_p0(mid)

        if did_hit:
            lo = mid   # still admissible; move right
        else:
            hi = mid   # overshoot; move left

        if hi - lo < 1e-8:
            break

    p0_star = lo
    print(f"Optimal p0 = F'(0) ≈ {p0_star:.10f}")
    return p0_star

# ---------------------------------------------------------
# helper for plot formatting
# ---------------------------------------------------------
def disable_offset(ax):
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== running revised nondegenerate shooting version ===")

    p0_star = find_optimal_p0()
    print(f"\nOptimal p0 = F'(0) ≈ {p0_star:.10f}")

    sol, _, did_hit = solve_path_given_p0(p0_star)
    if not did_hit:
        raise RuntimeError("Final solution did not hit F=F0.")

    if sol.t_events[0].size == 0:
        raise RuntimeError("Final solution did not hit F=F0.")

    Wgp = float(sol.t_events[0][0])
    Fgp, pgp = sol.y_events[0][0]

    print(f"Wgp ≈ {Wgp:.10f}")
    print(f"Value matching: F(Wgp)-F0(Wgp)  = {Fgp - F0(Wgp):.6e}")
    print(f"Smooth pasting: p(Wgp)-F0'(Wgp) = {pgp - F0p(Wgp):.6e}")

    W = sol.t
    F = sol.y[0]
    p = sol.y[1]

    mask = W <= Wgp + 1e-12
    W = W[mask]
    F = F[mask]
    p = p[mask]

    if abs(W[-1] - Wgp) > 1e-12:
        W = np.append(W, Wgp)
        F = np.append(F, Fgp)
        p = np.append(p, pgp)

    q_vals = np.zeros_like(W)
    a_vals = np.zeros_like(W)
    c_vals = np.zeros_like(W)
    drift_vals = np.zeros_like(W)

    for i in range(len(W)):
        q_vals[i], a_vals[i] = q_and_a_star(W[i], F[i], p[i])
        c_vals[i] = c_star_from_p(p[i])
        drift_vals[i] = r * (W[i] - u(c_vals[i]) + h(a_vals[i]))

    # W* = max of F
    idx_Wstar = np.argmax(F)
    W_star = W[idx_Wstar]

    # W** = first point where p becomes negative (for this sqrt utility case)
    neg_idx = np.where(p < 0)[0]
    W_ss = W[neg_idx[0]] if len(neg_idx) > 0 else np.nan

    print(f"W*   ≈ {W_star:.10f}")
    print(f"W**  ≈ {W_ss:.10f}")
    print(f"min(F-F0) on path = {np.min(F - F0(W)):.6e}")
    print(f"max effort        = {np.max(a_vals):.6e}")
    print(f"min effort        = {np.min(a_vals):.6e}")
    print(f"max consumption   = {np.max(c_vals):.6e}")
    print(f"min drift         = {np.min(drift_vals):.6e}")
    print(f"max drift         = {np.max(drift_vals):.6e}")

    if np.max(a_vals) >= A_MAX - 1e-4:
        print("WARNING: effort is hitting A_MAX; increase A_MAX and rerun.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Profit
    axes[0, 0].plot(W, F, label='F(W)')
    axes[0, 0].plot(W, F0(W), '--', label='F0(W)')
    axes[0, 0].axvline(W_star, linestyle=':', label='W*')
    if np.isfinite(W_ss):
        axes[0, 0].axvline(W_ss, linestyle='--', label='W**')
    axes[0, 0].axvline(Wgp, linestyle='-.', label='Wgp')
    axes[0, 0].set_title('Profit')
    axes[0, 0].set_xlabel('W')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    disable_offset(axes[0, 0])

    # Effort
    axes[0, 1].plot(W, a_vals)
    axes[0, 1].axvline(W_star, linestyle=':')
    if np.isfinite(W_ss):
        axes[0, 1].axvline(W_ss, linestyle='--')
    axes[0, 1].axvline(Wgp, linestyle='-.')
    axes[0, 1].set_title('Effort a(W)')
    axes[0, 1].set_xlabel('W')
    axes[0, 1].grid(alpha=0.3)
    disable_offset(axes[0, 1])

    # Consumption
    axes[1, 0].plot(W, c_vals)
    axes[1, 0].axvline(W_star, linestyle=':')
    if np.isfinite(W_ss):
        axes[1, 0].axvline(W_ss, linestyle='--')
    axes[1, 0].axvline(Wgp, linestyle='-.')
    axes[1, 0].set_title('Consumption c(W)')
    axes[1, 0].set_xlabel('W')
    axes[1, 0].grid(alpha=0.3)
    disable_offset(axes[1, 0])

    # Drift
    axes[1, 1].plot(W, drift_vals)
    axes[1, 1].axvline(W_star, linestyle=':')
    if np.isfinite(W_ss):
        axes[1, 1].axvline(W_ss, linestyle='--')
    axes[1, 1].axvline(Wgp, linestyle='-.')
    axes[1, 1].set_title('Drift of W')
    axes[1, 1].set_xlabel('W')
    axes[1, 1].grid(alpha=0.3)
    disable_offset(axes[1, 1])

    plt.tight_layout()
    plt.show()