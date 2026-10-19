import numpy as np

from caldis.solvers.steady import solve_steady
from caldis.solvers.newton import newton
from caldis.experiments.results import Results
from caldis.core.dof import preflight 

def _consistent_init(system, states, tol):
    """Index-1 DAE consistent initialization at t=0: hold the differential states at
    their initial condition, let their time-derivatives be free, and solve the algebraic
    variables so EVERY residual holds. Avoids recording unsolved start values for the
    algebraic quantities (mdot, W, Q...) at t=0. Non-fatal: on failure, leaves the raw
    start (the first real step then corrects it)."""
    unk = system.unknowns
    n = len(unk)
    sidx = [unk.index(v) for v in states]
    x_ic = np.array([unk[i].value for i in sidx], dtype=float)

    def G0(z):
        system.set_x(z[:n])
        for k, v in enumerate(states):
            v.der = z[n + k]
        R = list(system.residuals())                       # n eqs (differential use der)
        R += [z[sidx[k]] - x_ic[k] for k in range(len(states))]   # pin states to the IC
        return np.array(R, dtype=float)

    z0 = np.empty(n + len(states))
    z0[:n] = [v.value for v in unk]
    z0[n:] = 0.0                                            # start the der guesses at 0
    sol = newton(G0, z0, ftol=tol, xtol=tol)
    if not sol.success:
        return False
    system.set_x(sol.x[:n])
    for k, v in enumerate(states):
        v.der = sol.x[n + k]
    return True


def simulate(system, t_end, dt, update_inputs=None, init="steady", tol=1e-8,
             progress=False):
    if not system._assembled:
        system.assemble()
    system.check()
    
    if update_inputs is not None:
        update_inputs(system, 0.0)    
    
    # same pre-solve guard as solve(): fail early & clearly instead of a singular
    # Jacobian mid-run. raise_on_fail=True because simulate has no return-None path.
    preflight(system, verbose=progress, raise_on_fail=True)

    states = [v for v in system.unknowns if v.differential]

    if progress and init == "steady":
        print("  simulate: initial steady solve (t=0)...", end="", flush=True)

    # if init == "steady":
    #     sol0 = solve_steady(system)
    #     if not sol0.success:
    #         if progress:
    #             print(" FAILED")
    #         raise RuntimeError(f"Initialisation stationnaire échouée : {sol0.message}")
    #     if progress and init == "steady":
    #         print(" done.")

    # for v in states:
    #     v.der = 0.0

    # system.residuals()
    # rec0 = system.collect_outputs()

    if init == "steady":
        sol0 = solve_steady(system)
        if not sol0.success:
            if progress:
                print(" FAILED")
            raise RuntimeError(f"Initialisation stationnaire échouée : {sol0.message}")
        if progress and init == "steady":
            print(" done.")

    for v in states:
        v.der = 0.0

    if init != "steady":
        _consistent_init(system, states, tol)      # algebraic vars consistent with the IC at t=0

    system.residuals()
    rec0 = system.collect_outputs()


    hist = {k: [v] for k, v in rec0.items()}
    times = [0.0]
    x_prev = {v.name: v.value for v in states}

    n_steps = int(round(t_end / dt))

    # --- optional progress display ---
    _bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(total=n_steps, desc="simulate", unit="step")
        except Exception:
            _bar = None            # tqdm absent -> fallback on a carriage-return line

    for k in range(1, n_steps + 1):
        t = k * dt
        if update_inputs is not None:
            update_inputs(system, t)

        def G(x):
            system.set_x(x)
            for v in states:
                v.der = (v.value - x_prev[v.name]) / dt
            return system.residuals()

        guess = np.array([v.value for v in system.unknowns], dtype=float)
        sol = newton(G, guess, ftol=tol, xtol=tol)
        if not sol.success:
            if _bar is not None:
                _bar.close()
            raise RuntimeError(f"Pas t={t:.4g}s : {sol.message}")
        system.set_x(sol.x)
        for v in states:
            v.der = (v.value - x_prev[v.name]) / dt

        system.residuals()
        for kk, val in system.collect_outputs().items():
            hist[kk].append(val)
        times.append(t)
        x_prev = {v.name: v.value for v in states}

        # --- progress update ---
        if _bar is not None:
            fs = getattr(sol, "res_scaled", None)
            _bar.update(1)
            _bar.set_postfix(t=f"{t:.0f}s",
                            **({"|F|": f"{fs:.0e}"} if fs is not None else {}))
        elif progress:
            fs = getattr(sol, "res_scaled", None)
            tail = f"  |F|={fs:.1e}" if fs is not None else ""
            print(f"\r  simulate  [{k:4d}/{n_steps}]  t={t:7.1f}s{tail}",
                  end="", flush=True)

    if _bar is not None:
        _bar.close()
    elif progress:
        print()

    return Results(np.array(times), {kk: np.array(vv)
                                     for kk, vv in hist.items()})