
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.core.dof import check_dof
from caldis.components.sources.boundary import FlowSource, ClosedEnd
from caldis.components.vessels.control_volume import ControlVolume
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.dynamic import simulate

FLUID = "Air"
be = CoolPropBackend(FLUID)

# etat initial du volume : 1 bar, 20 °C
p_init, T_init = 1.0e5, 293.15
h_init = PropsSI("H", "P", p_init, "T", T_init, FLUID)

# alimentation : air a 20 °C, debit constant
mdot = 1.0e-3
h_in = PropsSI("H", "P", p_init, "T", T_init, FLUID)   # meme T que le volume au depart

V = 0.01                                               # 10 L rigides

sys_ = System("remplissage")
src = FlowSource("src", mdot=mdot, h=h_in)
vol = ControlVolume("vol", backend=be, V=V, p_init=p_init, h_init=h_init, Qdot=0.0)
cap = ClosedEnd("cap")

sys_.add(src, vol, cap)
sys_.connect(src.out, vol.inl)
sys_.connect(vol.out, cap.inl)          # sortie bouchee

print(check_dof(sys_))                   # doit etre carre

# pas de regime permanent -> init=None (part de l'etat construit) ; entrees constantes
t, hist = simulate(sys_, t_end=60.0, dt=0.5, update_inputs=None, init=None)

p = hist["vol.p"]; h = hist["vol.h"]; M = hist["vol.M"]; U = hist["vol.U"]
T = np.array([be.T(p[i], h[i]) for i in range(len(t))])
rho = M / V

print(f"\n{FLUID} — remplissage d'un volume ferme de {V*1e3:.0f} L, "
      f"mdot = {mdot*1e3:.1f} g/s\n")
print(f"{'t(s)':>5s} {'p(bar)':>9s} {'T(°C)':>8s} {'M(g)':>8s} {'rho':>8s}")
for i in range(0, len(t), len(t) // 10):
    print(f"{t[i]:5.0f} {p[i]/1e5:9.4f} {T[i]-273.15:8.3f} {M[i]*1e3:8.3f} {rho[i]:8.3f}")

# --- validations -------------------------------------------------------------
# 1. p et T croissent strictement
assert np.all(np.diff(p) > 0), "la pression doit croitre"
assert np.all(np.diff(T) > 0), "la temperature doit croitre"

# 2. le remplissage adiabatique CHAUFFE le gaz : T finale > T alimentation (20 °C)
print(f"\nT alimentation = {T_init-273.15:.2f} °C ; T finale = {T[-1]-273.15:.2f} °C")
assert T[-1] > T_init + 1.0, "le remplissage adiabatique doit rechauffer le gaz"

# 3. conservation EXACTE (Euler implicite exact sur dM/dt et dU/dt constants)
M_exp = M[0] + mdot * t[-1]
U_exp = U[0] + mdot * h_in * t[-1]
print(f"M final = {M[-1]*1e3:.4f} g (attendu {M_exp*1e3:.4f})")
print(f"U final = {U[-1]:.2f} J (attendu {U_exp:.2f})")
assert abs(M[-1] - M_exp) < 1e-9 * M_exp, "masse non conservee"
assert abs(U[-1] - U_exp) < 1e-9 * abs(U_exp), "energie non conservee"

print("\nOK : p et T montent, remplissage rechauffe le gaz, masse et energie conservees.")