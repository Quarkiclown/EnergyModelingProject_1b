from caldis.solvers.newton import newton


def solve_steady(system, tol=1e-8, max_iter=40):
    """Régime permanent : Newton amorti mis à l'échelle (solvers/newton.py), der=0 forcé.
    TODO : jacobienne AD (CasADi), bornage explicite, homotopie."""
    if not system._assembled:
        system.assemble()
    for v in system.unknowns:
        v.der = 0.0

    def f(x):
        system.set_x(x)
        return system.residuals()

    sol = newton(f, system.x0(), scale=system.scales(), bounds=system.bounds(),
                 ftol=tol, xtol=tol, max_iter=max_iter)
    system.set_x(sol.x)
    return sol