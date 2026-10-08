from caldis.core.dof import preflight        # au lieu de check_dof, check_startpoint
from caldis.solvers.homotopy import solve_homotopy


def solve(system, steps=3, verbose=True, show_homotopy=False,
          diagnostics=True, raise_on_fail=False):
    if not system._assembled:
        system.assemble()

    # Stages 1-3 (DOF + start evaluable + sanity) -> shared preflight
    if not preflight(system, verbose=verbose, raise_on_fail=raise_on_fail):
        return None

    # Stage 4: solve (expensive)
    sol = solve_homotopy(system, steps=steps, verbose=show_homotopy)

    if sol.success:
        if verbose:
            fs = getattr(sol, "res_scaled", None)
            acc = f"|F_scaled|={fs:.1e}" if fs is not None else "converged"
            print(f"✓ SOLVED ({acc})")
    else:
        if verbose:
            print(f"✗ FAILED: {sol.message}")
        if diagnostics:
            from caldis.core.diagnostics import convergence_report, jacobian_report
            convergence_report(system, sol)
            if "singul" in sol.message.lower():
                jacobian_report(system)
            if verbose:
                print("  -> re-anchor the variable(s) named above (set_start_port), "
                      "then solve again.")
        if raise_on_fail:
            raise RuntimeError(f"solve failed: {sol.message}")
    
    # ensure the system state always reflects sol.x, so a user's manual
    # inspection after solve() matches what the report showed (success or fail).
    # ... après la résolution et le convergence_report éventuel ...
    system.set_x(sol.x)                      # state reflects the last solution
    for c in system.components:
        _update_outputs_recursive(c)
    return sol


def _update_outputs_recursive(comp):
    """Fill outputs on a component and its children (composites)."""
    try:
        comp.update_outputs()
    except Exception:
        pass                                 # a degenerate stop state may be uncomputable
    for child in comp.children:
        _update_outputs_recursive(child)
