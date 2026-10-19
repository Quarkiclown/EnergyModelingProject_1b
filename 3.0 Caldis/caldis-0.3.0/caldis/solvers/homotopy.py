import numpy as np
from scipy.optimize import OptimizeResult

from caldis.solvers.newton import newton


def solve_homotopy(system, steps=3, tol=1e-8, dlam_min=1e-3, verbose=True):
    """Global Newton homotopy: R_lambda(x) = R(x) - (1-lambda) * R(x0).
    Fully generic; deforms the whole residual vector so constraints and boundaries
    are relaxed automatically. Bounded Newton at each step, warm-started; a failed
    step is halved. A final polish Newton at lambda=1 tightens the residual."""
    if not system._assembled:
        system.assemble()
    x = system.x0()
    system.set_x(x)
    R0 = np.asarray(system.residuals(), dtype=float).copy()
    lo, hi = system.bounds()
    sc = system.scales()

    def solve_at(lmbda, guess):
        def F(xx):
            system.set_x(xx)
            return system.residuals() - (1.0 - lmbda) * R0
        return newton(F, guess, scale=sc, bounds=(lo, hi), ftol=tol, xtol=tol)

    lam = 0.0
    dlam = 1.0 / steps
    n_solves = 0
    while lam < 1.0 - 1e-9:
        lam_try = min(lam + dlam, 1.0)
        sol = solve_at(lam_try, x)
        n_solves += 1
        if sol.success:
            x = sol.x
            lam = lam_try
            if verbose:
                fscaled = getattr(sol, "res_scaled", np.max(np.abs(sol.fun)))
                print(f"  lambda={lam:6.4f}  ok  ({sol.nit:2d} it, "
                      f"|F_scaled|={fscaled:.1e})")
            dlam = min(dlam * 1.5, 1.0 / steps)
        else:
            dlam *= 0.5
            if verbose:
                print(f"  lambda={lam_try:6.4f}  FAIL -> subdivide (dlam={dlam:.4f})")
            if dlam < dlam_min:
                system.set_x(x)
                return OptimizeResult(x=x, fun=sol.fun, success=False, nit=n_solves,
                    message=f"homotopy stuck at lambda={lam:.4f} "
                            f"(dlam<{dlam_min}): {sol.message}")

    # --- final polish at lambda=1 (tighten residual to ftol) ---
    def Ffull(xx):
        system.set_x(xx)
        return system.residuals()
    polish = newton(Ffull, x, scale=sc, bounds=(lo, hi), ftol=tol, xtol=tol)
    if polish.success:
        x = polish.x
    system.set_x(x)
    fscaled = getattr(polish, "res_scaled", np.max(np.abs(polish.fun)))
    return OptimizeResult(x=x, fun=system.residuals(), success=True, nit=n_solves + 1,
                          res_scaled=fscaled,
                          message=f"homotopy converged ({n_solves} steps + polish, "
                                  f"|F_scaled|inf={fscaled:.1e})")