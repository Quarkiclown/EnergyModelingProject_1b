"""Test autonome du backend CoolProp — auto-vérifiant.

On DÉRIVE les points de test depuis CoolProp lui-même (h de saturation à p fixée),
donc pas besoin de connaître par cœur les propriétés du fluide : le test reste valable
quel que soit le fluide choisi.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CoolProp.CoolProp import PropsSI
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.fluids.state import FluidState

FLUID = "R290"          # propane ; essaie aussi "R32", "R410A"
p = 5.0e5               # 5 bar

be = CoolPropBackend(FLUID)

# états de saturation à cette pression
h_liq = PropsSI("H", "P", p, "Q", 0.0, FLUID)
h_vap = PropsSI("H", "P", p, "Q", 1.0, FLUID)
T_sat = PropsSI("T", "P", p, "Q", 0.0, FLUID)

points = {
    "sous-refroidi": h_liq - 20_000.0,
    "diphasique x~0.5": 0.5 * (h_liq + h_vap),
    "surchauffé": h_vap + 20_000.0,
}

print(f"Fluide {FLUID} à {p/1e5:.1f} bar — T_sat = {T_sat-273.15:.2f} °C\n")
print(f"{'régime':18s} {'T (°C)':>9s} {'rho':>9s} {'x':>7s}")
for label, h in points.items():
    st = FluidState(be, p, h)
    print(f"{label:18s} {st.T-273.15:9.2f} {st.rho:9.2f} {st.x:7.2f}")

# vérifications
st2 = FluidState(be, p, points["diphasique x~0.5"])
assert abs(st2.x - 0.5) < 1e-3, "titre diphasique incorrect"
assert abs(st2.T - T_sat) < 1e-6, "T diphasique doit valoir T_sat"
assert FluidState(be, p, points["sous-refroidi"]).x == -1.0
assert FluidState(be, p, points["surchauffé"]).x == 2.0
print("\nOK : les 3 régimes sont correctement identifiés.")