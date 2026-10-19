from caldis.core.component import Component


class FluidBoundary(Component):
    """Fluid boundary condition, declared by named arguments.

    Each state property (p, T, h) and mdot is imposed if given, left FREE if None.
    Convention: an argument you provide is imposed, one you omit is solved for.
    State rule: at most TWO of (p, T, h) (two properties fix a state; three raises).
    Imposing T or h requires a backend. Each imposed value adds one residual.

    Examples:
        FluidBoundary("src", p=3e5, T=313, mdot=-2, backend=be)   # all imposed
        FluidBoundary("snk", p=1e5, backend=be)                   # p only
        FluidBoundary("in",  T=313, mdot=-2, backend=be)          # p is free
    """

    def __init__(self, name, p=None, T=None, h=None, mdot=None, backend=None):
        super().__init__(name)
        n_state = sum(x is not None for x in (p, T, h))
        if n_state > 2:
            raise ValueError(f"{name}: at most two of (p, T, h) may be imposed "
                             f"(got {n_state}) -- over-specified state")
        if (T is not None or h is not None) and backend is None:
            raise ValueError(f"{name}: imposing T or h requires a backend")
        self.backend = backend
        self.params["specs"] = {k: v for k, v in
                                (("p", p), ("T", T), ("h", h), ("mdot", mdot))
                                if v is not None}
        self.port = self.add_port("port", backend=backend)

        # fluid-aware starts (write on the Variable OBJECTS, not the value properties)
        s = self.params["specs"]
        if "p" in s:
            self.port._p.start = s["p"]
        if "mdot" in s:
            self.port._mdot.start = s["mdot"]
        if "h" in s:
            self.port._h.start = s["h"]
        elif "T" in s and "p" in s:
            self.port._h.start = backend.h_pt(s["p"], s["T"])

    def residuals(self):
        s = self.params["specs"]
        p, m, h = self.port.p, self.port.mdot, self.port.h
        r = []
        if "p" in s:
            r.append(p - s["p"])
        if "mdot" in s:
            r.append(m - s["mdot"])
        if "h" in s:
            r.append(h - s["h"])
        if "T" in s:
            # robust: backend on known-good (p, T_target), not on current h
            r.append(self.backend.h_pt(p, s["T"]) - h)
        return r

    def residual_names(self):
        return [k for k in ("p", "mdot", "h", "T") if k in self.params["specs"]]

    def update_outputs(self):
        self._outputs = {}      # boundary exposes its state via port.p / port.T directly