"""Per-element context passed to every physics brick of the discretized exchanger.

One ElementContext describes a single control volume of a single channel. The skeleton
builds it in two stages and then hands it to the bricks:

  Stage 1 (void-free): state, transport properties and geometry are filled.
  Stage 2 (void-dependent): the void brick sets `eps`, then the skeleton computes the
           hold-up density, the in-situ velocity and the Reynolds number.

A brick receives this object, READS the fields it needs and RETURNS a value. It must not
mutate the context (the only exception is the void brick, which produces `eps`). See the
brick code contract for the full rules.

STRICT ACCESS: fields that are computed later start as a sentinel. Reading one before it
is filled raises AttributeError instead of silently returning None -- this catches a
brick that reads a field from a stage it does not have access to (e.g. the void brick
reading `Re` or `rho_eff`, which only exist after the void fraction is known).

Quality convention (field `x`): 0 <= x <= 1 is two-phase; x = -1 flags subcooled liquid;
x = 2 flags superheated vapour. Bricks must handle all three.
"""

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Optional


class _Unset:
    """Sentinel marking a context field not yet filled by the skeleton."""
    __slots__ = ()
    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


@dataclass(repr=False)
class ElementContext:
    # ---- identity + state, set at construction (always available) ----
    side: str = ""                 # "hot" or "cold"
    index: int = 0                 # element index along the channel (0 = first cell)
    p: float = 0.0                 # pressure [Pa]
    h: float = 0.0                 # specific enthalpy [J/kg]
    mdot: float = 0.0              # mass flow magnitude through the element [kg/s]
    T_wall: float = 0.0            # wall temperature of this element [K]
    R_fouling: float = 0.0         # fouling resistance [m2.K/W]
    params: dict = field(default_factory=dict)  # component params (m_wall, cp_wall, ...)

    # ---- local state derived in stage 1 ----
    x: float = _UNSET              # vapour quality [-]  (0..1 two-phase; -1 liq; 2 vap)
    T: float = _UNSET              # temperature [K]

    # ---- fluid properties (stage 1) ----
    rho: float = _UNSET            # bulk density at (p, h) [kg/m3]
    rho_g: float = _UNSET          # saturated vapour density at p [kg/m3]
    rho_l: float = _UNSET          # saturated liquid density at p [kg/m3]
    h_g: float = _UNSET            # saturated vapour enthalpy at p [J/kg]
    h_l: float = _UNSET            # saturated liquid enthalpy at p [J/kg]
    mu: float = _UNSET             # dynamic viscosity [Pa.s]
    k: float = _UNSET              # thermal conductivity [W/m.K]
    cp: float = _UNSET             # specific heat at constant pressure [J/kg.K]
    Pr: float = _UNSET             # Prandtl number [-]
    sigma: float = _UNSET          # surface tension [N/m]

    # ---- geometry of the element (stage 1, from the geometry brick) ----
    A_exchange: float = _UNSET     # heat-exchange area of the element [m2]
    A_cross: float = _UNSET        # flow cross-section area [m2]
    V_internal: float = _UNSET     # internal fluid volume of the element [m3]
    dh: float = _UNSET             # hydraulic diameter [m]
    perimeter_wet: float = _UNSET  # wetted perimeter [m]
    L_elem: float = _UNSET         # element length along the flow [m]
    factor_geometry: float = _UNSET  # dimensionless geometry/enhancement factor
    fin_length: float = _UNSET     # fin length [m]
    fin_thickness: float = _UNSET  # fin thickness [m]
    fin_perimeter: float = _UNSET  # fin perimeter [m]
    fin_k: float = _UNSET          # fin material conductivity [W/m.K]

    # ---- flow quantities (stage 1) ----
    G: float = _UNSET              # mass flux, mdot / A_cross [kg/m2.s]
    j_g: float = _UNSET            # superficial vapour velocity [m/s]
    j_l: float = _UNSET            # superficial liquid velocity [m/s]

    # ---- signed face flow (set by the skeleton before the pressure brick) ----
    mdot_face: float = _UNSET      # signed mass flow on the element's face [kg/s]

    # ---- void-dependent quantities (stage 2) ----
    eps: float = _UNSET            # vapour void fraction [-]  (produced by void brick)
    rho_eff: float = _UNSET        # hold-up (effective) density [kg/m3]
    rho_h_eff: float = _UNSET      # hold-up rho*h product [J/m3]
    v: float = _UNSET              # in-situ velocity [m/s]
    Re: float = _UNSET             # Reynolds number [-]

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if value is _UNSET:
            raise AttributeError(
                f"ElementContext.{name!r} is not available yet: it is filled later by "
                f"the skeleton. A brick reading it here is accessing a field outside the "
                f"stage it runs in (e.g. a void brick must not read eps/rho_eff/v/Re)."
            )
        return value

    def __repr__(self):
        shown = []
        for f in fields(self):
            v = object.__getattribute__(self, f.name)
            shown.append(f"{f.name}={'<unset>' if v is _UNSET else v!r}")
        return f"ElementContext({', '.join(shown)})"