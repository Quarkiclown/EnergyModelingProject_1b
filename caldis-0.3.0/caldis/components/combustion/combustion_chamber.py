from caldis.core.component import Component


class CombustionChamber(Component):
    """Simplified gas-turbine combustion chamber -- "air + fuel-flow" level.

    Real MASS balance (m_gas = m_air + m_fuel) and ENERGY balance (fuel LHV heats
    the stream), but NO chemistry: the gas keeps AIR properties downstream. Each
    stream's enthalpy is taken relative to its own reference state (T_ref, p_ref),
    so the two backends' arbitrary CoolProp offsets cancel exactly.

    Ports (3): air_in (Air), fuel_in (fuel backend), gas_out (Air). mdot > 0 enters,
    so gas_out.mdot < 0.

    Residuals (3): mass; pressure p_gas = p_air*(1-dp_rel); energy balance.

    Parameters (defaults for methane): LHV, eta_comb, dp_rel, AFR_stoich (post-proc),
    T_ref/p_ref (LHV reference), T_max_warn (material limit, warning only).
    """

    def __init__(self, name, backend_air, backend_fuel,
                 LHV=50.0e6, eta_comb=1.0, dp_rel=0.04,
                 AFR_stoich=17.2, T_ref=298.15, p_ref=101325.0,
                 T_max_warn=1700.0):
        super().__init__(name)
        self.backend_air = backend_air
        self.backend_fuel = backend_fuel
        self.params.update(LHV=LHV, eta_comb=eta_comb, dp_rel=dp_rel,
                           AFR_stoich=AFR_stoich, T_ref=T_ref, p_ref=p_ref,
                           T_max_warn=T_max_warn)
        self.air_in  = self.add_port("air_in",  backend=backend_air)
        self.fuel_in = self.add_port("fuel_in", backend=backend_fuel)
        self.gas_out = self.add_port("gas_out", backend=backend_air)
        # constant reference enthalpies (each in its own backend's frame)
        self._h_a_ref = backend_air.h_pt(p_ref, T_ref)
        self._h_f_ref = backend_fuel.h_pt(p_ref, T_ref)

    def residuals(self):
        eta = self.params["eta_comb"]
        LHV = self.params["LHV"]
        dp_rel = self.params["dp_rel"]
        m_air, m_fuel, m_gas = self.air_in.mdot, self.fuel_in.mdot, self.gas_out.mdot
        h_a, h_f, h_out = self.air_in.h, self.fuel_in.h, self.gas_out.h
        return [
            m_air + m_fuel + m_gas,                                    # 1 mass
            self.gas_out.p - self.air_in.p * (1 - dp_rel),             # 2 pressure
            (m_air * (h_a - self._h_a_ref)                             # 3 energy
             + m_fuel * (h_f - self._h_f_ref + eta * LHV)
             + m_gas * (h_out - self._h_a_ref)),
        ]

    def residual_names(self):
        return ["mass", "pressure", "energy"]

    def update_outputs(self):
        m_fuel = self.fuel_in.mdot
        LHV = self.params["LHV"]
        afr = self.air_in.mdot / m_fuel if abs(m_fuel) > 1e-30 else float("inf")
        self._outputs = {
            "Qdot_fuel": m_fuel * LHV,
            "Qdot_released": m_fuel * LHV * self.params["eta_comb"],
            "AFR": afr,
            "excess_air": afr / self.params["AFR_stoich"],
        }

    @property
    def Qdot_fuel(self): return self._outputs["Qdot_fuel"]
    @property
    def Qdot_released(self): return self._outputs["Qdot_released"]
    @property
    def AFR(self): return self._outputs["AFR"]
    @property
    def excess_air(self): return self._outputs["excess_air"]

    def _own_start_sanity(self):
        m_air, m_fuel = self.air_in.mdot, self.fuel_in.mdot
        T_air = self.air_in.T
        T_out = self.gas_out.T
        afr = m_air / m_fuel if abs(m_fuel) > 1e-30 else float("inf")
        lam = afr / self.params["AFR_stoich"]
        return [
            (m_air > 0, f"{self.path}: air inlet flow must be > 0 (got {m_air:.3g})"),
            (m_fuel > 0, f"{self.path}: fuel inlet flow must be > 0 (got {m_fuel:.3g})"),
            (T_out > T_air,
             f"{self.path}: outlet ({T_out-273.15:.0f} °C) should be hotter than "
             f"air inlet ({T_air-273.15:.0f} °C)"),
            (lam > 1.0,
             f"{self.path}: mixture is rich (lambda={lam:.2f} <= 1); expected lean"),
            (T_out < self.params["T_max_warn"],
             f"{self.path}: outlet {T_out:.0f} K exceeds material warn limit "
             f"{self.params['T_max_warn']:.0f} K"),
        ]