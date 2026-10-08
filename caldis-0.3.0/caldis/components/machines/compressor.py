from caldis.core.component import Component


class Compressor(Component):
    """Adiabatic compressor with constant isentropic efficiency.
    eta_is DIVIDES the isentropic enthalpy rise: h_out = h_in + (h_out_s - h_in)/eta.
    Wdot is the power ABSORBED, reported NEGATIVE (work done ON the fluid;
    sign convention: work produced by the system > 0)."""

    def __init__(self, name, backend, eta_is):
        super().__init__(name)
        self.backend = backend
        self.params.update(eta_is=eta_is)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)

    def _h_out(self):
        """Discharge enthalpy from the isentropic efficiency (shared by residuals)."""
        be = self.backend
        eta = self.params["eta_is"]
        s_in = be.s(self.inl.p, self.inl.h)
        h_out_is = be.h_ps(self.out.p, s_in)
        return self.inl.h + (h_out_is - self.inl.h) / eta

    def residuals(self):
        return [
            self.inl.mdot + self.out.mdot,   # mass
            self.out.h - self._h_out(),      # isentropic
        ]

    def residual_names(self):
        return ["mass", "isentropic"]

    def update_outputs(self):
        self._outputs = {
            "Wdot": -self.inl.mdot * (self.out.h - self.inl.h),   # < 0: absorbed
            "pressure_ratio": self.out.p / self.inl.p,
        }

    @property
    def Wdot(self): return self._outputs["Wdot"]
    @property
    def pressure_ratio(self): return self._outputs["pressure_ratio"]

    def _own_start_sanity(self):
        p_in, p_out = self.inl.p, self.out.p
        return [(p_out > p_in,
                 f"{self.path}: p_out ({p_out/1e5:.2f} bar) must be > p_in "
                 f"({p_in/1e5:.2f} bar) for compression")]

    def exergy_balance(self, T0, p0):
        X_in  = self.inl.exergy_flow(T0, p0)
        X_out = self.out.exergy_flow(T0, p0)
        Wdot_absorbed = abs(self.inl.mdot * (self.out.h - self.inl.h))
        return self._finish_balance(T0, p0, Wdot_absorbed, X_out - X_in)