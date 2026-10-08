import numpy as np


import numpy as np


def jacobian_report(system, top=6):
    """Diagnose a singular / ill-conditioned Jacobian at the current point.

    The Jacobian is balanced on BOTH sides (columns by variable scale, rows by
    residual magnitude) so entries are O(1). This removes unit artefacts: a small
    variable like mdot no longer shows up as 'weak' just because it is small. After
    balancing, a genuinely weak row = an equation with no grip; a genuinely weak
    column = a variable no equation constrains -> re-anchor its start.
    Generic: needs no model knowledge.
    """
    from caldis.solvers.newton import _fd_jacobian
    x = system.x0(); system.set_x(x)
    F = np.asarray(system.residuals(), float)
    s = system.scales()
    J = _fd_jacobian(lambda xx: (system.set_x(xx), system.residuals())[1], x, F, s)

    # --- balance both sides so healthy entries are O(1) ---
    Js = J * s                                        # column scaling (variables)
    row_mag = np.maximum(np.max(np.abs(Js), axis=1), 1e-30)
    Jb = Js / row_mag[:, None]                        # row scaling (equations)
    row_norm = np.max(np.abs(Jb), axis=1)             # ~1 for healthy equations
    col_norm = np.max(np.abs(Jb), axis=0)             # small only if genuinely weak

    labels = system.residual_labels()
    var_names = [v.name for v in system.unknowns]

    rank = np.linalg.matrix_rank(Jb)
    full = min(J.shape)

    rank = np.linalg.matrix_rank(Jb)
    full = min(J.shape)
    print("\n=== Jacobian structure report ===")
    print(f"shape {J.shape} | rank {rank} (full = {full})")

    if rank == full:
        print("  -> full rank: non-singular here. Weak rows/cols below are benign.")
        print(f"\nWeakest equations:")
        for i in np.argsort(row_norm)[:top]:
            lbl = labels[i] if i < len(labels) else f"res[{i}]"
            print(f"  {row_norm[i]:.2e}   {lbl}")
        print(f"\nWeakest variables:")
        for j in np.argsort(col_norm)[:top]:
            print(f"  {col_norm[j]:.2e}   {var_names[j]}")
    else:
        print(f"  -> RANK DEFICIENT by {full - rank}. The null-space below names the "
              "degenerate combination -- re-anchor the START of these variables.")
        U, sv, Vt = np.linalg.svd(Jb)
        print(f"\n  smallest singular values: {np.array2string(sv[-3:], precision=2)}")
        vnull = np.abs(Vt[-1])                      # right null vector -> variables
        print("\n  >> VARIABLES to re-anchor (start likely wrong):")
        for j in np.argsort(-vnull)[:top]:
            if vnull[j] > 1e-3:
                cur = system.unknowns[j].start
                print(f"     weight {vnull[j]:.2f}   {var_names[j]}   (current start={cur:.4g})")
        unull = np.abs(U[:, -1])                    # left null vector -> equations
        print("\n  >> EQUATIONS that are redundant/unconstrained:")
        for i in np.argsort(-unull)[:top]:
            if unull[i] > 1e-3:
                lbl = labels[i] if i < len(labels) else f"res[{i}]"
                print(f"     weight {unull[i]:.2f}   {lbl}")


def convergence_report(system, sol, top=8):

    """Post-mortem for a failed/converged solve. Residuals are RESCALED per-equation
    (by their Jacobian row magnitude) so violations are comparable across units --
    a mass balance and an energy balance are ranked on equal footing, not by raw
    magnitude (which would always favor large-scale enthalpy/pressure equations)."""
    from caldis.solvers.newton import _fd_jacobian
    system.set_x(sol.x)
    R = np.asarray(system.residuals(), dtype=float)
    labels = system.residual_labels()
    s = system.scales()

    # per-equation scale = magnitude of its balanced Jacobian row
    J = _fd_jacobian(lambda xx: (system.set_x(xx), system.residuals())[1], sol.x, R, s)
    row_mag = np.maximum(np.max(np.abs(J * s), axis=1), 1e-30)
    R_scaled = R / row_mag                       # adimensioned residual

    print(f"\n=== convergence report ({'OK' if sol.success else 'FAILED'}) ===")
    print(f"message: {sol.message}")
    print(f"|R_scaled|inf = {np.max(np.abs(R_scaled)):.3e}   "
          f"(|R_raw|inf = {np.max(np.abs(R)):.3e})\n")

    order = np.argsort(-np.abs(R_scaled))
    print(f"Top {top} violated equations (by SCALED residual):")

    for i in order[:top]:
        lbl = labels[i] if i < len(labels) else f"res[{i}]"
        print(f"  scaled={np.abs(R_scaled[i]):9.2e}  (raw={R[i]:+.2e})   {lbl}")
        detail = _explain_residual(system, lbl)
        if detail:
            print(detail)

    lo, hi = system.bounds()
    print("\nVariables at a bound:")
    any_b = False
    for j, v in enumerate(system.unknowns):
        if np.isfinite(lo[j]) and abs(sol.x[j]-lo[j]) < 1e-8*max(1,abs(lo[j])):
            print(f"  {v.name} at lower {lo[j]:.4g}"); any_b = True
        if np.isfinite(hi[j]) and abs(sol.x[j]-hi[j]) < 1e-8*max(1,abs(hi[j])):
            print(f"  {v.name} at upper {hi[j]:.4g}"); any_b = True
    if not any_b:
        print("  (none)")
    return R


def _explain_residual(system, label):
    if not label.startswith("conn "):
        return None
    # the label ends with ".p" / ".h" / ".mdot" -> that's the variable
    if label.endswith(".p"):    kind = "p"
    elif label.endswith(".h"):   kind = "h"
    elif label.endswith(".mdot"): kind = "mdot"
    else: return None
    for conn in system.connections:
        a = f"{conn.a.owner.name}.{conn.a.name}"
        b = f"{conn.b.owner.name}.{conn.b.name}"
        if a in label and b in label:
            va, vb = getattr(conn.a, kind), getattr(conn.b, kind)   # déjà des floats
            if kind == "p":
                return f"      {a}.p = {va/1e5:.3f} bar,  {b}.p = {vb/1e5:.3f} bar"
            if kind == "h":
                return f"      {a}.h = {va/1e3:.1f} kJ/kg,  {b}.h = {vb/1e3:.1f} kJ/kg"
            if kind == "mdot":
                return f"      {a}.mdot = {va:.4g} kg/s,  {b}.mdot = {vb:.4g} kg/s"
            return None
    return None