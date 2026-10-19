from CoolProp.CoolProp import PropsSI, PhaseSI

from caldis.fluids.backend import FluidBackend


class CoolPropBackend(FluidBackend):
    def __init__(self, fluid):
        self.fluid = fluid

    def T(self, p, h):
        return PropsSI("T", "P", p, "H", h, self.fluid)

    def rho(self, p, h):
        return PropsSI("D", "P", p, "H", h, self.fluid)

    def quality(self, p, h):
        phase = PhaseSI("P", p, "H", h, self.fluid)
        if phase == "twophase":
            return PropsSI("Q", "P", p, "H", h, self.fluid)
        if "liquid" in phase:
            return -1.0
        return 2.0

    # --- ajouts pour compresseur / échangeur ---
    def s(self, p, h):
        return PropsSI("S", "P", p, "H", h, self.fluid)      # entropie depuis (p, h)

    def h_ps(self, p, s):
        return PropsSI("H", "P", p, "S", s, self.fluid)      # h depuis (p, s) — isentropique

    def h_pt(self, p, T):
        return PropsSI("H", "P", p, "T", T, self.fluid)      # h depuis (p, T)

    def cp_secant(self, p, T1, T2):
        """cp sécant entre T1 et T2 à p donné : (h2 - h1)/(T2 - T1).
        Robuste aux propriétés variables ; -> cp local si T1~T2."""
        if abs(T2 - T1) < 1e-6:
            from CoolProp.CoolProp import PropsSI
            return PropsSI("CPMASS", "P", p, "T", T1, self.fluid)
        return (self.h_pt(p, T2) - self.h_pt(p, T1)) / (T2 - T1)
    
    def h_pq(self, p, Q):
        """Enthalpie à saturation : Q=1 vapeur saturée, Q=0 liquide saturé."""
        from CoolProp.CoolProp import PropsSI
        return PropsSI("H", "P", p, "Q", Q, self.fluid)

    def default_state(self, T_hint=283.15):
        """Universal, always-evaluable single-phase start: 1 bar at T_hint.
        NOT near the operating point for high-pressure refrigerant -- that is what
        level-2 hints (set_start with real p, T per region) are for."""
        p = 1e5
        return p, self.h_pt(p, T_hint)


    def mu(self, p, h):
        """Dynamic viscosity [Pa.s]."""
        try:
            return PropsSI("V", "P", p, "H", h, self.fluid)
        except Exception:
            return PropsSI("V", "P", p, "Q", 0.0, self.fluid)   # two-phase fallback

    def cond(self, p, h):
        """Thermal conductivity [W/m/K]."""
        try:
            return PropsSI("L", "P", p, "H", h, self.fluid)
        except Exception:
            return PropsSI("L", "P", p, "Q", 0.0, self.fluid)

    def cp(self, p, h):
        """Specific heat cp [J/kg/K]."""
        try:
            return PropsSI("C", "P", p, "H", h, self.fluid)
        except Exception:
            return PropsSI("C", "P", p, "Q", 0.0, self.fluid)

    def Pr(self, p, h):
        """Prandtl number [-] = cp*mu/k."""
        k = self.cond(p, h)
        if k and k == k and abs(k) > 1e-30:
            return self.cp(p, h) * self.mu(p, h) / k
        return float("nan")

    def sigma(self, p, h):
        """Surface tension [N/m] (saturation value; 0 outside the dome)."""
        try:
            return PropsSI("I", "P", p, "Q", 0.0, self.fluid)
        except Exception:
            return 0.0