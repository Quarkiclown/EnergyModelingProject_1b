from caldis.core.component import Component
from caldis.components.vessels.control_volume import ControlVolume
from caldis.components.vessels.thermal_mass import ThermalMass


class TankConstV(Component):
    """Composite: fluid volume (ControlVolume) + wall (ThermalMass), coupled by
    Q = UA_wall*(T_wall - T_fluid). A heater Q_heater is applied to the wall.
    The tank's fluid ports ARE the inner volume's ports (aliases). The coupling is
    injected into the children via their Qdot (computed in _own_residuals, re-
    evaluated each iteration -> captured by the Jacobian)."""

    def __init__(self, name, backend, V, p_init, h_init,
                 C_wall, T_wall_init, UA_wall, Q_heater=0.0):
        super().__init__(name)
        self.backend = backend
        self.params.update(UA_wall=UA_wall, Q_heater=Q_heater)
        self.fluid = self.add_child(
            ControlVolume("fluid", backend=backend, V=V,
                          p_init=p_init, h_init=h_init, Qdot=0.0))
        self.wall = self.add_child(
            ThermalMass("wall", C=C_wall, UA=0.0,
                        T_init=T_wall_init, T_env=T_wall_init, Qdot_ext=0.0))
        self.inl = self.fluid.inl        # alias: tank exposes the volume's ports
        self.out = self.fluid.out

    def _own_residuals(self):
        UA = self.params["UA_wall"]
        Qh = self.params["Q_heater"]
        T_f = self.backend.T(self.fluid.p, self.fluid.h)
        T_w = self.wall.T
        Q = UA * (T_w - T_f)                          # wall -> fluid
        self._Q_coupling = Q                          # kept for update_outputs
        self.fluid.params["Qdot"] = Q                 # fluid receives +Q
        self.wall.params["Qdot_ext"] = Qh - Q         # wall: heater - Q released
        return []

    def update_outputs(self):
        self._outputs = {
            "Q": getattr(self, "_Q_coupling", 0.0),
            "T_fluid": self.backend.T(self.fluid.p, self.fluid.h),
            "T_wall": self.wall.T,
        }

    @property
    def Q(self): return self._outputs["Q"]
    @property
    def T_fluid(self): return self._outputs["T_fluid"]
    @property
    def T_wall(self): return self._outputs["T_wall"]


class ControlVolumeP(Component):
    """0-D two-phase control volume at CONSTANT PRESSURE (variable volume).

    Counterpart of ControlVolume: there V is fixed and p is algebraic; here p is a
    fixed PARAMETER and V becomes the algebraic unknown (the volume breathes to keep the
    pressure constant). States: M (mass), U (internal energy). Algebraic: V, h. Qdot
    prescribed (param, possibly driven by a composite parent)."""

    def __init__(self, name, backend, p, h_init, V_init, Qdot=0.0):
        super().__init__(name)
        self.backend = backend
        self.params.update(p=p, Qdot=Qdot)
        rho0 = backend.rho(p, h_init)
        M0 = V_init * rho0
        U0 = V_init * (rho0 * h_init - p)
        self._M = self.add_variable("M", start=M0, differential=True)
        self._U = self.add_variable("U", start=U0, differential=True)
        self._V = self.add_variable("V", start=V_init, lower=0.0)
        self._h = self.add_variable("h", start=h_init)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)
        for port in (self.inl, self.out):
            port._p.start = p
            port._h.start = h_init

    # public value access
    @property
    def M(self): return self._M.value
    @property
    def U(self): return self._U.value
    @property
    def V(self): return self._V.value
    @property
    def h(self): return self._h.value
    @property
    def p(self): return self.params["p"]          # imposed, constant

    def residuals(self):
        p = self.params["p"]
        Qdot = self.params["Qdot"]
        h, V = self._h.value, self._V.value
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
        p, h = self.params["p"], self._h.value
        self._outputs = {
            "rho": self.backend.rho(p, h),
            "T": self.backend.T(p, h),
            "x": self.backend.quality(p, h),
            "V": self._V.value,
        }

    @property
    def rho(self): return self._outputs["rho"]
    @property
    def T(self): return self._outputs["T"]
    @property
    def x(self): return self._outputs["x"]
    @property
    def V_out(self): return self._outputs["V"]


class TankConstP(Component):
    """Composite constant-pressure (variable-volume) storage: fluid volume
    (ControlVolumeP) + wall (ThermalMass), coupled by Q = UA_wall*(T_wall - T_fluid).
    A heater Q_heater is applied to the wall. The tank's fluid ports ARE the inner
    volume's ports (aliases). Same structure as the constant-volume Tank; only the inner
    volume differs (pressure imposed, volume free).

    Exergy: a storage device has no steady-state 'product', so exergy_balance returns
    None (no X_supplied / X_destroyed / eta_II). Stored exergy, if needed, is a separate
    state quantity to be added later."""

    def __init__(self, name, backend, p, h_init, V_init,
                 C_wall, T_wall_init, UA_wall, Q_heater=0.0):
        super().__init__(name)
        self.backend = backend
        self.params.update(UA_wall=UA_wall, Q_heater=Q_heater)
        self.fluid = self.add_child(
            ControlVolumeP("fluid", backend=backend, p=p,
                           h_init=h_init, V_init=V_init, Qdot=0.0))
        self.wall = self.add_child(
            ThermalMass("wall", C=C_wall, UA=0.0,
                        T_init=T_wall_init, T_env=T_wall_init, Qdot_ext=0.0))
        self.inl = self.fluid.inl        # alias: tank exposes the volume's ports
        self.out = self.fluid.out

    def _own_residuals(self):
        UA = self.params["UA_wall"]
        Qh = self.params["Q_heater"]
        T_f = self.backend.T(self.fluid.p, self.fluid.h)
        T_w = self.wall.T
        Q = UA * (T_w - T_f)                          # wall -> fluid
        self._Q_coupling = Q                          # kept for update_outputs
        self.fluid.params["Qdot"] = Q                 # fluid receives +Q
        self.wall.params["Qdot_ext"] = Qh - Q         # wall: heater - Q released
        return []

    def update_outputs(self):
        self._outputs = {
            "Q": getattr(self, "_Q_coupling", 0.0),
            "T_fluid": self.backend.T(self.fluid.p, self.fluid.h),
            "T_wall": self.wall.T,
            "V": self.fluid.V,
        }

    @property
    def Q(self): return self._outputs["Q"]
    @property
    def T_fluid(self): return self._outputs["T_fluid"]
    @property
    def T_wall(self): return self._outputs["T_wall"]
    @property
    def V(self): return self._outputs["V"]

    def exergy_balance(self, T0, p0):
        return None    # storage device: no steady-state product -> no eta_II