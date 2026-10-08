from caldis.core.component import Component


class ThermalMass(Component):
    """0-D thermal mass: C dT/dt = UA(T_env - T) + Qdot_ext.
    Qdot_ext (default 0) is the external heat input driven by a composite parent."""

    def __init__(self, name, C, UA, T_init, T_env=None, Qdot_ext=0.0):
        super().__init__(name)
        self.params.update(C=C, UA=UA,
                           T_env=(T_env if T_env is not None else T_init),
                           Qdot_ext=Qdot_ext)
        self._T = self.add_variable("T", start=T_init, differential=True)

    @property
    def T(self): return self._T.value

    def residuals(self):
        C = self.params["C"]
        UA = self.params["UA"]
        T_env = self.params["T_env"]
        Qext = self.params["Qdot_ext"]
        return [C * self._T.der - (UA * (T_env - self._T.value) + Qext)]

    def residual_names(self):
        return ["energy"]

    def update_outputs(self):
        self._outputs = {"Qdot_ext": self.params["Qdot_ext"]}