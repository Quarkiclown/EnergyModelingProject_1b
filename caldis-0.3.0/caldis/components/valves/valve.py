"""Valve / expansion device: pressure-drop element, adiabatic (isenthalpic).

Modes (by Kv): Kv given -> simulation (flow set by the flow law); Kv=None -> sizing
(Kv is an unknown, the system must impose an extra condition, e.g. a flow).
Flow laws (by flow_law): "quadratic" mdot=Kv*sign(dp)*sqrt(|dp|) (orifice), or
"linear" mdot=Kv*dp (smoother, for convergence trouble). Kv units differ per law.
Residuals (3): mass, flow law, isenthalpic.
"""

import math

from caldis.core.component import Component


class Valve(Component):
    def __init__(self, name, Kv=None, backend=None, flow_law="quadratic", eps=1e-9):
        super().__init__(name)
        self.backend = backend
        if flow_law not in ("quadratic", "linear"):
            raise ValueError(f"flow_law must be 'quadratic' or 'linear', got {flow_law!r}")
        self.params.update(flow_law=flow_law, eps=eps)
        self.inl = self.add_port("inl", backend=backend)
        self.out = self.add_port("out", backend=backend)

        # Kv given -> parameter (simulation); Kv None -> unknown (sizing).
        self.sizing = (Kv is None)
        if self.sizing:
            self._Kv_var = self.add_variable("Kv", start=1e-4, lower=0.0, scale=1e-4)
        else:
            self.params["Kv"] = Kv

    def _Kv(self):
        """Current Kv value (variable in sizing mode, parameter otherwise)."""
        return self._Kv_var.value if self.sizing else self.params["Kv"]

    def residuals(self):
        dp = self.inl.p - self.out.p
        Kv = self._Kv()
        if self.params["flow_law"] == "quadratic":
            m_law = Kv * math.copysign(math.sqrt(abs(dp) + self.params["eps"]), dp)
        else:
            m_law = Kv * dp
        return [
            self.inl.mdot + self.out.mdot,   # mass
            self.inl.mdot - m_law,           # flow law
            self.out.h - self.inl.h,         # isenthalpic
        ]

    def residual_names(self):
        return ["mass", "flow_law", "isenthalpic"]

    def update_outputs(self):
        self._outputs = {
            "Kv": self._Kv(),
            "dp": self.inl.p - self.out.p,
        }

    @property
    def Kv(self): return self._outputs["Kv"]
    @property
    def dp(self): return self._outputs["dp"]

    def _own_start_sanity(self):
        p_in, p_out = self.inl.p, self.out.p
        return [
            (p_in >= p_out,
             f"{self.path}: p_in ({p_in/1e5:.2f} bar) should be >= p_out "
             f"({p_out/1e5:.2f} bar) for a pressure-dropping valve"),
        ]

    def exergy_balance(self, T0, p0):
        # purely dissipative: nothing recovered, all supplied exergy is destroyed
        Xd = T0 * self.entropy_generation()
        return self._finish_balance(T0, p0, Xd, 0.0)