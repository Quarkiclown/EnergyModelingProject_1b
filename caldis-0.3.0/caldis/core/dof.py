"""Vérificateur de degrés de liberté.

C'est LA réponse directe à la douleur d'origine (les "système structurellement
singulier" incompréhensibles de Dymola). Avant toute résolution, on compte inconnues
vs équations et on rend un diagnostic explicite au niveau système.

Version squelette : bilan global (compte). Étapes suivantes prévues :
  - appariement structurel (matching biparti) pour DÉSIGNER la/les variable(s)
    non appariée(s) — "il manque une condition sur mdot du compresseur", plutôt qu'un
    simple compte ;
  - détection de sur-spécification locale (une même variable fixée deux fois).
"""


class DOFReport:
    def __init__(self, n_unknowns, n_equations):
        self.n_unknowns = n_unknowns
        self.n_equations = n_equations
        self.dof = n_unknowns - n_equations

    @property
    def ok(self):
        return self.dof == 0

    def __str__(self):
        base = f"{self.n_unknowns} inconnues, {self.n_equations} équations"
        if self.dof == 0:
            return f"[DOF ok] {base} — système carré."
        if self.dof > 0:
            return (f"[DOF sous-spécifié] {base} : il manque {self.dof} "
                    f"condition(s) aux limites (variables non déterminées).")
        return (f"[DOF sur-spécifié] {base} : {-self.dof} équation(s) de trop "
                f"(condition imposée en double ?).")


def check_dof(system):
    if not system._assembled:
        system.assemble()
    report = DOFReport(system.n_unknowns(), system.n_equations())
    return report

def check_startpoint(system):
    """Check that the start point (.start) is evaluable, and pinpoint the exact
    failing residual (by label) and the port states feeding it -- not just which
    component. Raises with a detailed, actionable message. Call before solving."""
    import numpy as np
    if not system._assembled:
        system.assemble()
    system.set_x(system.x0())
    problems = []

    # build a label map: for each component, the index range of its residuals
    labels = system.residual_labels()

    for c in system.components:
        try:
            r = np.asarray(c.residuals(), dtype=float)
        except Exception as e:
            problems.append(f"component {c.path}: raised {type(e).__name__}: {e}\n"
                            f"{_port_dump(c)}")
            continue
        bad = np.where(~np.isfinite(r))[0]
        if len(bad):
            # name each offending residual if the component exposes residual_names()
            names = getattr(c, "residual_names", lambda: None)()
            for i in bad:
                rname = names[i] if names and i < len(names) else f"residual[{i}]"
                problems.append(f"component {c.path}: {rname} is non-finite "
                                f"(value={r[i]})\n{_port_dump(c)}")

    for conn in system.connections:
        try:
            rc = np.asarray(conn.residuals(), dtype=float)
            if not np.all(np.isfinite(rc)):
                problems.append(f"connection {conn}: non-finite")
        except Exception as e:
            problems.append(f"connection {conn}: {type(e).__name__}: {e}")

    for name, fn in system.constraints:
        try:
            v = float(fn())
            if not np.isfinite(v):
                problems.append(f"constraint {name!r}: non-finite result")
        except Exception as e:
            problems.append(f"constraint {name or '?'}: {type(e).__name__}: {e}")

    if problems:
        raise ValueError("Start point not evaluable:\n  - " + "\n  - ".join(problems))
    return True


def _port_dump(comp):
    """Readable dump of a component's port start states, to diagnose why a residual
    is non-finite (out-of-domain (p,h), zero flow, equal temperatures, ...)."""
    lines = []
    for port in getattr(comp, "ports", []):
        p, h, m = port.p, port.h, port.mdot
        be = getattr(port, "backend", None)
        Tinfo = ""
        if be is not None:
            try:
                T = be.T(p, h)
                Tinfo = f", T={T-273.15:.1f}°C"
            except Exception:
                Tinfo = ", T=<out of domain>"
        lines.append(f"      {port.name:8s} p={p/1e5:.3f}bar  h={h/1e3:.1f}kJ/kg  "
                     f"mdot={m:+.4g}{Tinfo}")
    return "\n".join(lines)


def _iter_ports(comp):
    ports = list(getattr(comp, "ports", []))
    for child in getattr(comp, "children", []):
        ports += _iter_ports(child)
    return ports


def _backend_for_port(comp, port):
    """Choisit le backend adapté au port : côté r_* -> backend_r, s_* -> backend_s,
    sinon backend générique."""
    name = port.name
    if name.startswith("r_") and getattr(comp, "backend_r", None):
        return comp.backend_r
    if name.startswith("s_") and getattr(comp, "backend_s", None):
        return comp.backend_s
    return getattr(comp, "backend", None)


def preflight(system, verbose=True, check_start=True, raise_on_fail=False):
    """Shared pre-solve checks (structural + start), used by both solve and simulate.

    Order: DOF (square?) -> start point evaluable? -> physical sanity (warnings only).
    Returns True if OK. On failure: prints diagnostics, then raises (raise_on_fail=True)
    or returns False.
    """
    if not system._assembled:
        system.assemble()

    dof = check_dof(system)
    if not dof.ok:
        if verbose:
            print(f"✗ DOF: {dof}")
            print("  -> the model is not square. Add/remove a constraint or boundary "
                  "spec (MODELLING issue, not a start issue).")
        if raise_on_fail:
            raise ValueError(str(dof))
        return False
    if verbose:
        print(f"✓ DOF: square ({dof.n_unknowns} unknowns)")

    if check_start:
        try:
            check_startpoint(system)
            if verbose:
                print("✓ start point: evaluable")
        except ValueError as e:
            if verbose:
                print("✗ start point NOT evaluable:")
                print("   " + str(e).replace("\n", "\n   "))
            if raise_on_fail:
                raise
            return False

    warnings = system.check_start_sanity(verbose=False)
    if verbose and warnings:
        print(f"⚠ {len(warnings)} start-sanity warning(s):")
        for w in warnings:
            print("   - " + w)
        print("  (non-blocking.)")
    return True


