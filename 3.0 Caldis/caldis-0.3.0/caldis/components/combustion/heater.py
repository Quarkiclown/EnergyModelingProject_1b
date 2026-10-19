
from caldis.core.component import Component

class SimpleHeater(Component):
    """Simplified heater / cooler for a single fluid stream (ports: inl, out).

    Adds a thermal duty Qdot to the stream: Qdot > 0 heats, Qdot < 0 cools.
    If Qdot is given at construction it is a PARAMETER; if left None it becomes an
    UNKNOWN solved by the system (e.g. to meet an outlet-temperature target imposed
    elsewhere). Only the energy balance is modelled; pressure drop is an optional
    constant (default 0).

    Exergy needs the heat-source temperature T_source (reservoir exchanging Qdot):
      - finite T_source     -> Carnot-quality heat,
      - T_source = float('inf') -> electric / work-grade heat,
      - T_source = None     -> exergy balance not computable (returns None).
    """

    def __init__(self, name, backend, Qdot=None, dp=0.0, T_source=None):
        super().__init__(name)
        self.backend = backend
        self.params.update(dp=dp)
        if T_source is not None:
            self.params.update(T_source=T_source)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)

        self._Qdot_fixed = Qdot is not None
        if self._Qdot_fixed:
            self.params.update(Qdot=Qdot)
            self._Qdot = None
        else:
            # unknown duty: the system must supply one closing equation (e.g. T_out)
            self._Qdot = self.add_variable("Qdot", start=0.0, scale=1e3)

    def _qdot(self):
        """Live duty value (parameter or unknown), for use inside residuals."""
        return self.params["Qdot"] if self._Qdot_fixed else self._Qdot.value

    def residuals(self):
        Qdot = self._qdot()
        return [
            self.inl.mdot + self.out.mdot,                        # mass
            self.out.p - (self.inl.p - self.params["dp"]),        # pressure drop
            self.inl.mdot * (self.out.h - self.inl.h) - Qdot,     # energy balance
        ]

    def residual_names(self):
        return ["mass", "dp", "energy"]

    def update_outputs(self):
        self._outputs = {"Qdot": self._qdot()}

    @property
    def Qdot(self):
        return self._outputs["Qdot"]

    def _own_start_sanity(self):
        return [
            (abs(self.inl.mdot) > 1e-12,
             f"{self.path}: inlet mass flow is ~0 -> outlet enthalpy undetermined"),
        ]

    def entropy_generation(self):
        """Sgen = mdot*(s_out - s_in) - Qdot/T_source. With T_source = inf (electric),
        the heat carries no entropy; without T_source, the heat term is dropped."""
        be = self.backend
        s_in  = be.s(self.inl.p, self.inl.h)
        s_out = be.s(self.out.p, self.out.h)
        T_source = self.params.get("T_source", None)
        q_over_T = (self._qdot() / T_source) if T_source is not None else 0.0
        return self.inl.mdot * (s_out - s_in) - q_over_T

    def exergy_balance(self, T0, p0):
        T_source = self.params.get("T_source", None)
        if T_source is None:
            return None    # heat quality undefined without a source temperature
        Qdot = self._qdot()
        X_Q = Qdot * (1.0 - T0 / T_source)                       # exergy carried by the heat
        dX = self.out.exergy_flow(T0, p0) - self.inl.exergy_flow(T0, p0)  # exergy gained by stream
        # supplied = exergy spent (the heat interaction), recovered = exergy added to the stream.
        # Same labelling for heating and for sub-ambient cooling; X_destroyed = X_Q - dX = T0*Sgen.
        X_supplied, X_recovered = X_Q, dX
        return self._finish_balance(T0, p0, X_supplied, X_recovered)