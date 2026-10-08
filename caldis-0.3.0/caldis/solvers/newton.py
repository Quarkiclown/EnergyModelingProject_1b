import numpy as np
from scipy.optimize import OptimizeResult


def _safe_eval(f, x):
    try:
        F = np.asarray(f(x), dtype=float)
    except Exception:
        return None
    if F.ndim != 1 or not np.all(np.isfinite(F)):
        return None
    return F


def _fd_jacobian(f, x, F0, s, rel=1e-7):
    n = x.size
    J = np.empty((F0.size, n))
    for j in range(n):
        step = rel * max(s[j], abs(x[j]), 1e-30)
        xp = x.copy(); xp[j] += step
        Fp = _safe_eval(f, xp)
        if Fp is None:
            xm = x.copy(); xm[j] -= step
            Fm = _safe_eval(f, xm)
            J[:, j] = 0.0 if Fm is None else (F0 - Fm) / step
        else:
            J[:, j] = (Fp - F0) / step
    return J


def _max_feasible_step(x, dx, lower, upper):
    alpha = 1.0
    for j in range(x.size):
        if dx[j] > 0 and np.isfinite(upper[j]):
            room = (upper[j] - x[j]) / dx[j]
            if room < alpha: alpha = max(room, 0.0)
        elif dx[j] < 0 and np.isfinite(lower[j]):
            room = (lower[j] - x[j]) / dx[j]
            if room < alpha: alpha = max(room, 0.0)
    return alpha


def newton(f, x0, scale=None, jac=None, bounds=None, ftol=1e-8, xtol=1e-8,
           max_iter=40, damping=True, margin=0.99):
    """Damped Newton with automatic scaling, row balancing and step projection
    into bounds. Returns OptimizeResult with .x, .fun (raw residual), .success,
    .message, .nit, and .res_scaled (the balanced |F|inf the convergence test uses).
    """
    x = np.array(x0, dtype=float)
    n = x.size
    if bounds is None:
        lower = np.full(n, -np.inf); upper = np.full(n, np.inf)
    else:
        lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        x = np.clip(x, lower, upper)

    base = np.abs(scale) if scale is not None else np.abs(x)
    s = np.where(base > 1e-30, base, 1.0)

    F = _safe_eval(f, x)
    if F is None:
        return OptimizeResult(x=x, fun=np.full(n, np.nan), success=False, nit=0,
                              res_scaled=np.inf, message="évaluation initiale hors domaine")

    Dr = np.ones_like(F)
    for it in range(1, max_iter + 1):
        J = jac(x) if jac is not None else _fd_jacobian(f, x, F, s)
        Jc = J * s
        row = np.maximum(np.max(np.abs(Jc), axis=1), 1e-30)
        Dr = 1.0 / row
        Fs = Dr * F
        res_scaled = float(np.max(np.abs(Fs)))
        if res_scaled < ftol:
            return OptimizeResult(x=x, fun=F, success=True, nit=it - 1,
                                  res_scaled=res_scaled,
                                  message=f"convergé (|F_scaled|inf < {ftol:g})")
        g0 = np.linalg.norm(Fs)

        A = Dr[:, None] * Jc
        try:
            dy = np.linalg.solve(A, -Fs)
        except np.linalg.LinAlgError:
            return OptimizeResult(x=x, fun=F, success=False, nit=it,
                                  res_scaled=res_scaled,
                                  message=f"jacobienne singulière (it {it})")
        dx = s * dy

        alpha_max = _max_feasible_step(x, dx, lower, upper) * margin
        if alpha_max <= 0.0:
            return OptimizeResult(x=x, fun=F, success=False, nit=it,
                                  res_scaled=res_scaled,
                                  message=f"variable au bord, aucun pas admissible (it {it})")

        lam = alpha_max
        x_new = F_new = lam_used = None
        while lam >= 1e-6 * alpha_max:
            xn = x + lam * dx
            Fn = _safe_eval(f, xn)
            if Fn is not None:
                gn = np.linalg.norm(Dr * Fn)
                if gn <= (1.0 - 1e-4 * lam) * g0:
                    x_new, F_new, lam_used = xn, Fn, lam
                    break
                if x_new is None:
                    x_new, F_new, lam_used = xn, Fn, lam
            lam *= 0.5

        if x_new is None:
            return OptimizeResult(x=x, fun=F, success=False, nit=it,
                                  res_scaled=res_scaled,
                                  message=f"pas hors domaine à toute échelle (it {it})")

        x, F = x_new, F_new
        if np.linalg.norm(dy * lam_used) < xtol:
            r = float(np.max(np.abs(Dr * F)))
            if r < ftol * 1e2:
                return OptimizeResult(x=x, fun=F, success=True, nit=it, res_scaled=r,
                                      message=f"convergé (|dx|<xtol, |F_scaled|={r:.1e})")
            return OptimizeResult(x=x, fun=F, success=False, nit=it, res_scaled=r,
                                  message=f"BLOQUÉ : pas minuscule mais |F_scaled|inf "
                                          f"= {r:.2e} (backtracking épuisé, "
                                          f"probablement mauvais point de départ)")

    r = float(np.max(np.abs(Dr * F)))
    return OptimizeResult(x=x, fun=F, success=False, nit=max_iter, res_scaled=r,
                          message=f"non-convergence ({max_iter} it, |F_scaled|inf = {r:.2e})")