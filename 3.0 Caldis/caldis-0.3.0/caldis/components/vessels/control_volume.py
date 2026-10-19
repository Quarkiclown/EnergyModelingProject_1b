from caldis.core.component import Component


class ControlVolume(Component):
    """0-D two-phase control volume. States: M (mass), U (internal energy). p, h are
    algebraic (via CoolProp). Qdot prescribed (param, possibly driven by a composite
    parent)."""

    def __init__(self, name, backend, V, p_init, h_init, Qdot=0.0):
        super().__init__(name)
        self.backend = backend
        self.params.update(V=V, Qdot=Qdot)
        rho0 = backend.rho(p_init, h_init)
        M0 = V * rho0
        U0 = V * (rho0 * h_init - p_init)
        self._M = self.add_variable("M", start=M0, differential=True)
        self._U = self.add_variable("U", start=U0, differential=True)
        self._p = self.add_variable("p", start=p_init, lower=0.0)
        self._h = self.add_variable("h", start=h_init)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)
        for port in (self.inl, self.out):
            port._p.start = p_init
            port._h.start = h_init

    # public value access
    @property
    def M(self): return self._M.value
    @property
    def U(self): return self._U.value
    @property
    def p(self): return self._p.value
    @property
    def h(self): return self._h.value

    def residuals(self):
        V = self.params["V"]
        Qdot = self.params["Qdot"]
        p, h = self._p.value, self._h.value
        rho = self.backend.rho(p, h)
        Hflow = self.inl.mdot * self.inl.h + self.out.mdot * self.out.h
        return [
            self._M.der - (self.inl.mdot + self.out.mdot),
            self._U.der - (Hflow + Qdot),
            self._M.value - V * rho,
            self._U.value - V * (rho * h - p),
            self.inl.p - p,
            self.out.p - p,
            self.out.h - h,
        ]

    def residual_names(self):
        return ["mass_dyn", "energy_dyn", "M_def", "U_def",
                "inl_p", "out_p", "out_h"]

    def update_outputs(self):
        p, h = self._p.value, self._h.value
        self._outputs = {
            "rho": self.backend.rho(p, h),
            "T": self.backend.T(p, h),
            "x": self.backend.quality(p, h),
        }

    @property
    def rho(self): return self._outputs["rho"]
    @property
    def T(self): return self._outputs["T"]
    @property
    def x(self): return self._outputs["x"]