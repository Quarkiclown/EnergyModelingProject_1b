from caldis.core.component import Component


class Turbine(Component):
    """Adiabatic turbine, constant isentropic efficiency. Counterpart of Compressor.

    eta_is MULTIPLIES the isentropic drop (compressor divides it):
        h_out = h_in - eta_is * (h_in - h_out_s).
    Wdot is the shaft power DELIVERED, reported POSITIVE. Net shaft power at system
    level = Wdot_turbine - Wdot_compressor. Pressures come from the system.

    Parameters: eta_is (isentropic eff.), eta_mech (mechanical eff., outputs only).
    """

    def __init__(self, name, backend, eta_is=0.89, eta_mech=1.0):
        super().__init__(name)
        self.backend = backend
        self.params.update(eta_is=eta_is, eta_mech=eta_mech)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)

    def _h_out_s(self):
        """Isentropic outlet enthalpy (shared by residuals and outputs)."""
        be = self.backend
        s_in = be.s(self.inl.p, self.inl.h)
        return be.h_ps(self.out.p, s_in)

    def residuals(self):
        h_in = self.inl.h
        h_out = h_in - self.params["eta_is"] * (h_in - self._h_out_s())   # multiply
        return [
            self.inl.mdot + self.out.mdot,   # mass
            self.out.h - h_out,              # expansion
        ]

    def residual_names(self):
        return ["mass", "isentropic"]

    def update_outputs(self):
        eta_m = self.params["eta_mech"]
        m = self.inl.mdot
        h_in, h_out = self.inl.h, self.out.h
        h_out_s = self._h_out_s()
        self._outputs = {
            "Wdot": eta_m * m * (h_in - h_out),          # > 0: delivered
            "Wdot_is": eta_m * m * (h_in - h_out_s),     # isentropic (max) power
            "pressure_ratio": self.inl.p / self.out.p,
            "h_out_s": h_out_s,
            "T_out_s": self.backend.T(self.out.p, h_out_s),
        }

    @property
    def Wdot(self): return self._outputs["Wdot"]
    @property
    def Wdot_is(self): return self._outputs["Wdot_is"]
    @property
    def pressure_ratio(self): return self._outputs["pressure_ratio"]
    @property
    def h_out_s(self): return self._outputs["h_out_s"]
    @property
    def T_out_s(self): return self._outputs["T_out_s"]

    def _own_start_sanity(self):
        be = self.backend
        p_in, p_out = self.inl.p, self.out.p
        m_in = self.inl.mdot
        T_in = be.T(p_in, self.inl.h)
        return [
            (p_out < p_in,
             f"{self.path}: p_out ({p_out/1e5:.2f} bar) must be < p_in "
             f"({p_in/1e5:.2f} bar) for expansion"),
            (m_in > 0,
             f"{self.path}: inlet flow must be > 0 (got {m_in:.3g})"),
            (T_in > 500.0,
             f"{self.path}: inlet {T_in:.0f} K seems cold for a gas turbine "
             f"(forgotten hot start?)"),
        ]

    def exergy_balance(self, T0, p0):
        X_in  = self.inl.exergy_flow(T0, p0)
        X_out = self.out.exergy_flow(T0, p0)
        Wdot = self.params["eta_mech"] * self.inl.mdot * (self.inl.h - self.out.h)
        return self._finish_balance(T0, p0, X_in - X_out, Wdot)