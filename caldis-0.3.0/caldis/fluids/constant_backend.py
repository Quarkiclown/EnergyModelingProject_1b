from caldis.fluids.backend import FluidBackend


class ConstantPropertiesBackend(FluidBackend):
    """PLACEHOLDER sans dépendance : fluide incompressible, cp constant.
    À remplacer par CoolPropBackend. Ne pas prendre les valeurs au sérieux."""

    def __init__(self, cp=4180.0, rho=1000.0, T_ref=273.15, h_ref=0.0):
        self.cp = cp
        self._rho = rho
        self.T_ref = T_ref
        self.h_ref = h_ref

    def T(self, p, h):
        return self.T_ref + (h - self.h_ref) / self.cp

    def rho(self, p, h):
        return self._rho

    def quality(self, p, h):
        return -1.0

    def h_pt(self, p, T):
        return self.h_ref + self.cp * (T - self.T_ref)

    def default_state(self, T_hint=283.15):
        return 1e5, self.h_pt(1e5, T_hint)
    