import numpy as np

class System:
    def __init__(self, name="system"):
        self.name = name
        self.components = []
        self.connections = []
        self.constraints = []          # callables () -> résidu (float), contraintes système
        self.unknowns = []
        self._assembled = False
        self._start_hints = []
        self._port_hints = []

    def add(self, *components):
        self.components.extend(components)
        self._assembled = False
        return components[0] if len(components) == 1 else components

    def connect(self, port_a, port_b, close_loop=False):
        from caldis.core.connection import Connection
        conn = Connection(port_a, port_b, close_loop=close_loop)
        self.connections.append(conn)
        self._assembled = False
        return conn

    def add_constraint(self, fn, name=""):
        if name and any(n == name for n, _ in self.constraints):
            raise ValueError(f"contrainte {name!r} déjà présente "
                             f"(utiliser replace_constraint pour l'écraser)")
        self.constraints.append((name, fn))
        self._assembled = False
        return fn

    def remove_constraint(self, name):
        n0 = len(self.constraints)
        self.constraints = [(n, f) for (n, f) in self.constraints if n != name]
        if len(self.constraints) == n0:
            raise KeyError(f"aucune contrainte nommée {name!r} à retirer "
                           f"(présentes : {[n for n, _ in self.constraints]})")
        self._assembled = False

    def replace_constraint(self, name, fn):
        """Retire puis rajoute : pratique pour changer une valeur cible."""
        self.remove_constraint(name)
        self.add_constraint(fn, name)

    def constraint_names(self):
        return [n for n, _ in self.constraints]

    def assemble(self):
        self.unknowns = [v for c in self.components for v in c.all_variables()]
        for i, v in enumerate(self.unknowns):
            v.index = i
        self._assembled = True
        return self

    def x0(self):
        return np.array([v.start for v in self.unknowns], dtype=float)

    def set_x(self, x):
        for v, xi in zip(self.unknowns, x):
            v.value = float(xi)

    def residuals(self):
        r = []
        for c in self.components:
            r += list(c.residuals())
        for conn in self.connections:
            r += list(conn.residuals())
        for _, fn in self.constraints:
            r.append(fn())
        return np.array(r, dtype=float)

    def n_unknowns(self):
        return len(self.unknowns)

    def n_equations(self):
        return len(self.residuals())

    def check(self):
        if not self._assembled:
            self.assemble()
        for c in self.components:
            c.check()

    def collect_outputs(self):
        into = {}
        for c in self.components:
            c.collect_outputs(into)
        return into

    # def scales(self):
    #     import numpy as np
    #     return np.array([v.scale if (v.scale and v.scale > 0) else 1.0
    #                      for v in self.unknowns], dtype=float)

    def scales(self):
        import numpy as np
        out = []
        for v in self.unknowns:
            if v.scale and v.scale > 0:
                out.append(v.scale)                     # scale explicite respecté
            else:
                out.append(max(abs(v.start), 1.0))      # auto : magnitude physique du start
        return np.array(out, dtype=float)
    
    def bounds(self):
        import numpy as np
        lo = np.array([v.lower if v.lower is not None else -np.inf for v in self.unknowns])
        hi = np.array([v.upper if v.upper is not None else  np.inf for v in self.unknowns])
        return lo, hi

    def set_start(self, component, p=None, T=None, mdot=None, side=None):
        """Level-2 deferred start hint (applied by init_starts, AFTER level-1).
        side: 'r' (r_* ports), 's' (s_* ports), None (all fluid ports of the component).
        p sets the pressure; T recomputes h via h_pt(p, T); mdot (magnitude) is signed
        per inlet(+)/outlet(-). Any argument left None keeps the level-1 value."""
        self._start_hints.append((component, p, T, mdot, side))
        return self

    def _apply_start_hint(self, comp, p, T, mdot, side):
        for port in comp.ports:
            name = port.name
            if side is not None and not name.startswith(side + "_"):
                continue
            be = port.backend
            if p is not None:
                port._p.start = p
            if T is not None and be is not None:
                port._h.start = be.h_pt(port._p.start, T)
            if mdot is not None:
                port._mdot.start = (-mdot if name.endswith("out") else mdot)

    def set_start_port(self, port, p=None, T=None, h=None, mdot=None):

        """Level-2 hint at PORT granularity (deferred, applied by init_starts after
        level-1). Needed for heat exchangers whose inlet/outlet are at different
        states. p sets pressure; T recomputes h via h_pt(p,T); h sets it directly;
        mdot is the signed start. None keeps the level-1 value."""
        self._port_hints.append((port, p, T, h, mdot))
        return self

    def check_start_sanity(self, verbose=True):
        """Level-2 diagnostic: check each component's physical preconditions at the
        current start point. Warns (does not raise) so you can still try to solve.
        Returns the list of failed checks."""
        if not self._assembled:
            self.assemble()
        self.set_x(self.x0())
        failed = []
        for c in self.components:
            for ok, msg in c.start_sanity():
                if not ok:
                    failed.append(msg)
        if verbose:
            if failed:
                print("Start sanity WARNINGS (start point may not converge):")
                for m in failed:
                    print("  ⚠ " + m)
            else:
                print("Start sanity: all component preconditions satisfied.")
        return failed

    def residual_labels(self):
        """Human label for each residual, in the SAME order as residuals().
        Lets diagnostics point at which equation is violated."""
        
        labels = []
        for c in self.components:
            n = len(c.residuals())
            names = getattr(c, "residual_names", lambda: None)()
            if names is not None and len(names) == n:
                labels += [f"{c.path}.{nm}" for nm in names]
            else:
                labels += [f"{c.path}[{i}]" for i in range(n)]

        for conn in self.connections:
            kinds = ["p", "h"] if conn.close_loop else ["p", "h", "mdot"]
            a = f"{conn.a.owner.name}.{conn.a.name}"
            b = f"{conn.b.owner.name}.{conn.b.name}"
            labels += [f"conn {a}.{k} = {b}.{k}" for k in kinds]

        for name, _ in self.constraints:
            labels.append(f"constraint:{name}")
        return labels

    def _all_ports(self):
        def walk(c):
            ports = [(c, p) for p in c.ports]
            for child in c.children:
                ports += walk(child)
            return ports
        out = []
        for c in self.components:
            out += walk(c)
        return out

    def port_table(self, as_dataframe=True):
        """Tabulate every fluid port after a solve. Two temperature columns to avoid
        unit confusion: T [K], T_C [°C]. Other columns in engineering units."""
        rows = []
        for comp, port in self._all_ports():
            be = port.backend
            T = s = x = rho = None
            if be is not None:
                try:
                    T = port.T
                    s = be.s(port._p.value, port._h.value)
                    x = be.quality(port._p.value, port._h.value)
                    rho = be.rho(port._p.value, port._h.value)
                except Exception:
                    pass
            rows.append({
                "component": comp.path, "port": port.name,
                "p_bar": port.p / 1e5,
                "T": T,
                "T_C": (None if T is None else T - 273.15),
                "h_kJkg": port.h / 1e3,
                "s_kJkgK": (None if s is None else s / 1e3),
                "x": x, "rho": rho, "mdot": port.mdot,
            })
        if as_dataframe:
            try:
                import pandas as pd
                return pd.DataFrame(rows).set_index(["component", "port"])
            except ImportError:
                pass
        return rows
    
    def init_starts(self, T_default=283.15, verbose=False):
        """Two-level fluid-aware start initialization.
        Level 1: fluid-aware default (p, h) on every port.
        Level 2a: per-side hints (set_start).  2b: per-port hints (set_start_port).
        Call after building the system, before solving."""
        if not self._assembled:
            self.assemble()

        # --- level 1: fluid-aware defaults on all ports (recurses into children) ---
        for c in self.components:
            c.apply_default_starts(T_default)

        # --- level 2a: per-side hints ---
        for (comp, p, T, mdot, side) in self._start_hints:
            self._apply_start_hint(comp, p, T, mdot, side)

        # --- level 2b: per-port hints (finest) ---
        for (port, p, T, h, mdot) in self._port_hints:
            be = port.backend
            if p is not None:
                port._p.start = p
            if h is not None:
                port._h.start = h
            elif T is not None:
                if be is None:
                    raise ValueError(f"set_start_port({port.path}, T=...) needs a "
                                     f"backend on the port, but none is attached")
                port._h.start = be.h_pt(port._p.start, T)
            if mdot is not None:
                port._mdot.start = mdot

        # --- level 3: internal starts (port->cell propagation + explicit overrides) ---
        for c in self.components:
            c.seed_internal_starts()
        
        # --- load starts into current values ---
        self.set_x(self.x0())
        if verbose:
            print(f"init_starts: level-1 + {len(self._start_hints)} side + "
                  f"{len(self._port_hints)} port hints")
        return self

    def structure(self, depth=None):

        """Print the structure of every top-level component in the system."""
        for c in self.components:
            c.structure(depth=depth)
            print()
        return None

    def set_reference(self, T_ref, p_ref=101325.0):
        """Dead state for exergy analysis (call before exergy_report)."""
        self.T_ref = T_ref
        self.p_ref = p_ref

    def exergy_report(self, T_ref=None, p_ref=None, as_dataframe=True):

        T0 = T_ref if T_ref is not None else getattr(self, "T_ref", None)
        p0 = p_ref if p_ref is not None else getattr(self, "p_ref", 101325.0)
        if T0 is None:
            raise ValueError("no reference temperature: call set_reference(T_ref) first")
        rows = []
        for c in self.components:
            bal = c.exergy_balance(T0, p0)     # computes AND writes onto c
            if bal is None:
                continue
            rows.append({"component": c.path, **bal})
        if as_dataframe:
            try:
                import pandas as pd
                return pd.DataFrame(rows).set_index("component")
            except ImportError:
                pass
        return rows

    def set_start_internal(self, component, base_name, value):
        """Set internal-variable starts of a component, by base name (one call for
        all elements): scalar = uniform, list = per-element profile. Symmetric with
        set_start_port. Call before init_starts."""
        component.set_internal_start(base_name, value)
        return self

    