import math

from caldis.core.component import Component


def _dtlm(dT1, dT2, name):
    """Log-mean temperature difference, sign-agnostic.
    dT1 == dT2 (equal, non-zero) is a normal case -> returns that value.
    Only dT1 == dT2 == 0 or a genuine cross (opposite signs) is ill-defined -> NaN."""
    a, b = abs(dT1), abs(dT2)
    if dT1 * dT2 < 0.0:
        return float("nan")
    if a < 1e-12 and b < 1e-12:
        return float("nan")
    if abs(a - b) < 1e-9:
        return max(a, b)
    return (a - b) / math.log(a / b)


class SizingHX(Component):

    """Generic single-zone counter-flow heat exchanger for sizing.

    ONE model for every heat exchanger (condenser, evaporator, recuperator,
    gas/gas, liquid/liquid), no 'role'. Single-zone LMTD; approximately valid even
    across a phase change, never fails on sign conventions.

    Channels 'hot'/'cold' are a documentation convention: the maths is sign-agnostic
    (a reversed hookup still computes), the names keep results readable and let
    start-sanity warn about a mis-wire.

    Ports (4): hot_in, hot_out (backend_hot); cold_in, cold_out (backend_cold).

    Parameters:
        h_hot, h_cold          : per-side CONVECTIVE (film) coefficients [W/m2/K].
        area_ratio_cold_to_hot : A_cold / A_hot (>=1 finned on the cold side, =1 plate).

    Overall coefficient (both films in series, wall neglected), referred to A_hot:
        U_overall_hot  = 1 / (1/h_hot + 1/(area_ratio*h_cold))
        A_hot          = UA / U_overall_hot
        A_cold         = area_ratio * A_hot ;  A_total = A_hot + A_cold
        U_overall_cold = U_overall_hot / area_ratio      (same UA, cold-side reference)

    Unknowns (2, solved by the system): UA, A_hot (read as hx.UA, hx.A_hot -> value).
    Derived outputs (after solve): U_overall_hot, U_overall_cold, A_cold, A_total,
        Qdot, DTLM.

    Residuals (7): 1-4 mass/dp per side; 5 coupling Qdot_hot + Qdot_cold = 0;
    6 UA = |Qdot_hot| / LMTD; 7 A_hot = UA / U_overall_hot. Pressures come from
    the system."""

    def __init__(self, name, backend_hot, backend_cold,
                 h_hot, h_cold, area_ratio_cold_to_hot):
        super().__init__(name)
        self.backend_hot = backend_hot
        self.backend_cold = backend_cold
        self.params.update(h_hot=h_hot, h_cold=h_cold,
                           area_ratio_cold_to_hot=area_ratio_cold_to_hot)
        self.hot_in   = self.add_port("hot_in",   backend=backend_hot)
        self.hot_out  = self.add_port("hot_out",  backend=backend_hot)
        self.cold_in  = self.add_port("cold_in",  backend=backend_cold)
        self.cold_out = self.add_port("cold_out", backend=backend_cold)
        self._UA    = self.add_variable("UA",    start=1000.0, lower=0.0, scale=1e3)
        self._A_hot = self.add_variable("A_hot", start=1.0,    lower=0.0, scale=1.0)

    # --- public value access for the unknowns ---
    @property
    def UA(self): return self._UA.value
    @property
    def A_hot(self): return self._A_hot.value

    def _U_overall_hot(self):
        """Overall heat-transfer coefficient referred to A_hot [W/m2/K].
        NOT a film coefficient: it combines both films in series."""
        p = self.params
        return 1.0 / (1.0 / p["h_hot"]
                      + 1.0 / (p["area_ratio_cold_to_hot"] * p["h_cold"]))

    def _compute(self):
        """Shared physics for residuals, outputs and sanity: the 4 port temperatures,
        the two heat rates and the LMTD. No side effect (stores nothing)."""
        bh, bc = self.backend_hot, self.backend_cold
        Th_in  = bh.T(self.hot_in.p,   self.hot_in.h)
        Th_out = bh.T(self.hot_out.p,  self.hot_out.h)
        Tc_in  = bc.T(self.cold_in.p,  self.cold_in.h)
        Tc_out = bc.T(self.cold_out.p, self.cold_out.h)
        Qdot_hot  = self.hot_in.mdot   * (self.hot_in.h  - self.hot_out.h)
        Qdot_cold = self.cold_out.mdot * (self.cold_out.h - self.cold_in.h)
        DTLM = _dtlm(Th_in - Tc_out, Th_out - Tc_in, self.path)
        return Th_in, Th_out, Tc_in, Tc_out, Qdot_hot, Qdot_cold, DTLM

    def residuals(self):
        U_overall_hot = self._U_overall_hot()
        _, _, _, _, Qdot_hot, Qdot_cold, DTLM = self._compute()
        return [
            self.hot_in.mdot + self.hot_out.mdot,       # 1 mass hot
            self.hot_out.p - self.hot_in.p,             # 2 dp hot
            self.cold_in.mdot + self.cold_out.mdot,     # 3 mass cold
            self.cold_out.p - self.cold_in.p,           # 4 dp cold
            Qdot_hot + Qdot_cold,                       # 5 coupling
            self._UA.value - abs(Qdot_hot) / DTLM,      # 6 UA def
            self._A_hot.value - self._UA.value / U_overall_hot,   # 7 A def
        ]

    def residual_names(self):
        return ["mass_hot", "dp_hot", "mass_cold", "dp_cold",
                "coupling_Q", "def_UA", "def_A"]

    def update_outputs(self):
        beta = self.params["area_ratio_cold_to_hot"]
        _, _, _, _, Qdot_hot, _, DTLM = self._compute()
        U_overall_hot = self._U_overall_hot()
        self._outputs = {
            "U_overall_hot": U_overall_hot,
            "U_overall_cold": U_overall_hot / beta,
            "A_cold": beta * self._A_hot.value,
            "A_total": (1.0 + beta) * self._A_hot.value,
            "Qdot": Qdot_hot,
            "DTLM": DTLM,
        }

    @property
    def U_overall_hot(self): return self._outputs["U_overall_hot"]
    @property
    def U_overall_cold(self): return self._outputs["U_overall_cold"]
    @property
    def A_cold(self): return self._outputs["A_cold"]
    @property
    def A_total(self): return self._outputs["A_total"]
    @property
    def Qdot(self): return self._outputs["Qdot"]
    @property
    def DTLM(self): return self._outputs["DTLM"]

    def _own_start_sanity(self):
        Th_in, Th_out, Tc_in, Tc_out, *_ = self._compute()
        dT1 = Th_in - Tc_out
        dT2 = Th_out - Tc_in
        return [
            (Th_in >= Tc_out and Th_out >= Tc_in,
             f"{self.path}: 'hot' channel is not hotter than 'cold' at start "
             f"(hot {Th_in-273.15:.0f}/{Th_out-273.15:.0f}°C vs "
             f"cold {Tc_in-273.15:.0f}/{Tc_out-273.15:.0f}°C) -- reversed hookup?"),
            (dT1 * dT2 > 0.0,
             f"{self.path}: temperature cross at start (dT1={dT1:.1f}, dT2={dT2:.1f} K)"),
            (abs(Th_in - Th_out) > 0.1,
             f"{self.path}: hot inlet/outlet nearly equal -> LMTD/flow degenerate"),
            (abs(Tc_out - Tc_in) > 0.1,
             f"{self.path}: cold inlet/outlet nearly equal -> cold MASS FLOW is "
             f"undetermined (anchor the cold-side inlet/outlet TEMPERATURES)"),
        ]

    def exergy_balance(self, T0, p0):

        Xh = self.hot_in.exergy_flow(T0, p0) - self.hot_out.exergy_flow(T0, p0)
        Xc = self.cold_out.exergy_flow(T0, p0) - self.cold_in.exergy_flow(T0, p0)
        return self._finish_balance(T0, p0, Xh, Xc)


def _effectiveness(NTU, Cr, flow_config):
    """Heat-exchanger effectiveness from NTU and the capacity-rate ratio Cr = Cmin/Cmax.

    Explicit (no log, no iteration): this is what makes the rating problem easy to
    converge, unlike the LMTD form used for sizing. Handles the removable singularity
    at Cr = 1 and the limit Cr -> 0 (one capacity rate much larger than the other)."""
    if NTU <= 0.0:
        return 0.0
    if Cr < 1e-9:                       # Cmax -> infinity (e.g. near a phase change)
        return 1.0 - math.exp(-NTU)
    if flow_config == "counterflow":
        if abs(Cr - 1.0) < 1e-6:        # removable singularity
            return NTU / (1.0 + NTU)
        e = math.exp(-NTU * (1.0 - Cr))
        return (1.0 - e) / (1.0 - Cr * e)
    # parallel / co-current
    return (1.0 - math.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)

class RatingHX(Component):
    """Single-zone heat exchanger in RATING (simulation) mode, effectiveness-NTU method.

    Counterpart of SizingHX: there UA/areas are unknown and sized from fixed duties;
    here UA is KNOWN (from areas + film coefficients, or given directly) and the OUTLET
    states are computed from the inlets. The effectiveness-NTU closure is explicit
    (NTU -> eps -> Q -> h_out), so it converges more easily than the LMTD form, which
    needs the log of outlet-dependent temperature differences.

    Channels 'hot'/'cold' are a naming convention; start-sanity warns about a mis-wire.
    Ports (4): hot_in, hot_out (backend_hot); cold_in, cold_out (backend_cold).

    SCOPE / VALIDITY: convection only -- no wall inertia, no radiation, no void fraction,
    pressure drop neglected (p_out = p_in). The effectiveness-NTU method assumes a
    CONSTANT specific heat per side (evaluated at the inlet), so this model is valid for
    SINGLE-PHASE (sensible) exchange only. It is NOT valid across a phase change
    (cp -> infinity inside the dome): use SizingHX (approximate LMTD) or the discretized
    HXSim for condensers/evaporators.

    UA parametrization -- give EXACTLY ONE of:
        - UA                                                   : overall conductance [W/K]
        - h_hot, h_cold, area_ratio_cold_to_hot, A_hot         : physical set
          UA = A_hot / (1/h_hot + 1/(area_ratio_cold_to_hot * h_cold))
    Passing both, or an incomplete physical set, raises a clear error.

    Residuals (6): mass + dp per side (4), hot and cold energy balances tied to the same
    duty Q = eps * Cmin * (T_hot_in - T_cold_in). No internal unknowns (UA is known);
    the outlet states are port variables solved by the system."""

    def __init__(self, name, backend_hot, backend_cold,
                 UA=None, h_hot=None, h_cold=None,
                 area_ratio_cold_to_hot=None, A_hot=None,
                 flow_config="counterflow"):
        super().__init__(name)
        if flow_config not in ("counterflow", "coflow"):
            raise ValueError(f"flow_config must be 'counterflow' or 'coflow', got {flow_config!r}")

        physical_args = {"h_hot": h_hot, "h_cold": h_cold,
                         "area_ratio_cold_to_hot": area_ratio_cold_to_hot, "A_hot": A_hot}
        has_physical = any(v is not None for v in physical_args.values())
        has_direct = UA is not None

        if has_direct and has_physical:
            raise ValueError(
                f"{name}: give EITHER UA, OR the physical set "
                f"(h_hot, h_cold, area_ratio_cold_to_hot, A_hot) -- not both.")
        if not has_direct and not has_physical:
            raise ValueError(
                f"{name}: missing conductance -- give UA, or the physical set "
                f"(h_hot, h_cold, area_ratio_cold_to_hot, A_hot).")
        if has_physical:
            missing = [k for k, v in physical_args.items() if v is None]
            if missing:
                raise ValueError(
                    f"{name}: the physical parametrization needs all of "
                    f"h_hot, h_cold, area_ratio_cold_to_hot, A_hot; missing: {missing}.")
            U_overall_hot = 1.0 / (1.0 / h_hot
                                   + 1.0 / (area_ratio_cold_to_hot * h_cold))
            UA = U_overall_hot * A_hot

        self.backend_hot = backend_hot
        self.backend_cold = backend_cold
        self.params.update(UA=UA, flow_config=flow_config)

        self.hot_in   = self.add_port("hot_in",   backend=backend_hot)
        self.hot_out  = self.add_port("hot_out",  backend=backend_hot)
        self.cold_in  = self.add_port("cold_in",  backend=backend_cold)
        self.cold_out = self.add_port("cold_out", backend=backend_cold)

    def _compute(self):
        """Shared physics: inlet temperatures, capacity rates, NTU, effectiveness, duty.
        Constant cp evaluated at each inlet (single-phase assumption). Stores nothing."""
        bh, bc = self.backend_hot, self.backend_cold
        Th_in = bh.T(self.hot_in.p,  self.hot_in.h)
        Tc_in = bc.T(self.cold_in.p, self.cold_in.h)
        cp_h = bh.cp(self.hot_in.p,  self.hot_in.h)
        cp_c = bc.cp(self.cold_in.p, self.cold_in.h)
        C_hot  = abs(self.hot_in.mdot)  * cp_h
        C_cold = abs(self.cold_in.mdot) * cp_c
        C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
        Cr = C_min / C_max if C_max > 1e-30 else 0.0
        NTU = self.params["UA"] / C_min if C_min > 1e-30 else 0.0
        eps = _effectiveness(NTU, Cr, self.params["flow_config"])
        Q = eps * C_min * (Th_in - Tc_in)
        return Th_in, Tc_in, C_hot, C_cold, Cr, NTU, eps, Q

    def residuals(self):
        *_, Q = self._compute()
        return [
            self.hot_in.mdot + self.hot_out.mdot,                         # 1 mass hot
            self.hot_out.p - self.hot_in.p,                               # 2 dp hot (none)
            self.cold_in.mdot + self.cold_out.mdot,                       # 3 mass cold
            self.cold_out.p - self.cold_in.p,                             # 4 dp cold (none)
            self.hot_in.mdot * (self.hot_in.h - self.hot_out.h) - Q,      # 5 hot energy
            self.cold_in.mdot * (self.cold_out.h - self.cold_in.h) - Q,   # 6 cold energy
        ]

    def residual_names(self):
        return ["mass_hot", "dp_hot", "mass_cold", "dp_cold",
                "energy_hot", "energy_cold"]

    def update_outputs(self):
        bh, bc = self.backend_hot, self.backend_cold
        Th_in, Tc_in, C_hot, C_cold, Cr, NTU, eps, Q = self._compute()
        self._outputs = {
            "Qdot": Q,
            "effectiveness": eps,
            "NTU": NTU,
            "Cr": Cr,
            "C_hot": C_hot,
            "C_cold": C_cold,
            "UA": self.params["UA"],
            "T_hot_out": bh.T(self.hot_out.p, self.hot_out.h),
            "T_cold_out": bc.T(self.cold_out.p, self.cold_out.h),
        }

    @property
    def Qdot(self): return self._outputs["Qdot"]
    @property
    def effectiveness(self): return self._outputs["effectiveness"]
    @property
    def NTU(self): return self._outputs["NTU"]
    @property
    def Cr(self): return self._outputs["Cr"]
    @property
    def C_hot(self): return self._outputs["C_hot"]
    @property
    def C_cold(self): return self._outputs["C_cold"]
    @property
    def UA(self): return self._outputs["UA"]
    @property
    def T_hot_out(self): return self._outputs["T_hot_out"]
    @property
    def T_cold_out(self): return self._outputs["T_cold_out"]

    def _own_start_sanity(self):
        bh, bc = self.backend_hot, self.backend_cold
        Th_in = bh.T(self.hot_in.p, self.hot_in.h)
        Tc_in = bc.T(self.cold_in.p, self.cold_in.h)
        return [
            (Th_in > Tc_in,
             f"{self.path}: hot inlet ({Th_in-273.15:.0f}°C) should be hotter than "
             f"cold inlet ({Tc_in-273.15:.0f}°C) -- reversed hookup?"),
            (abs(self.hot_in.mdot) > 1e-12,
             f"{self.path}: hot inlet mass flow is ~0 -> capacity rate / NTU undefined"),
            (abs(self.cold_in.mdot) > 1e-12,
             f"{self.path}: cold inlet mass flow is ~0 -> capacity rate / NTU undefined"),
        ]

    def exergy_balance(self, T0, p0):
        Xh = self.hot_in.exergy_flow(T0, p0) - self.hot_out.exergy_flow(T0, p0)
        Xc = self.cold_out.exergy_flow(T0, p0) - self.cold_in.exergy_flow(T0, p0)
        return self._finish_balance(T0, p0, Xh, Xc)



