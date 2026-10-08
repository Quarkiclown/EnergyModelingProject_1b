"""Validation de l'intégrateur : relaxation exponentielle d'une masse thermique.

Auto-vérifiant : on compare le transitoire numérique à la solution analytique
T(t) = T_inf + (T0 - T_inf) exp(-t/tau), tau = C/UA.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from caldis.core.system import System
from caldis.components.vessels.thermal_mass import ThermalMass
from caldis.solvers.dynamic import simulate

C, UA = 5000.0, 50.0        # J/K, W/K
tau = C / UA                # 100 s
T0, T_inf = 20.0, 80.0

sys_ = System("relaxation")
mass = ThermalMass("mass", C=C, UA=UA, T_init=T0, T_env=T0)  # départ à l'équilibre
sys_.add(mass)

def update_inputs(system, t):
    # échelon de température d'environnement à t=0+
    mass.params["T_env"] = T_inf if t > 0.0 else T0

dt, t_end = 1.0, 500.0
t, hist = simulate(sys_, t_end=t_end, dt=dt, update_inputs=update_inputs)

T_num = hist["mass.T"]
T_exact = T_inf + (T0 - T_inf) * np.exp(-t / tau)

print(f"tau = {tau:.0f} s, dt = {dt:.0f} s")
print(f"{'t (s)':>6s} {'T_num':>9s} {'T_exact':>9s}")
for i in range(0, len(t), 50):
    print(f"{t[i]:6.0f} {T_num[i]:9.3f} {T_exact[i]:9.3f}")

print(f"\nValeur finale : {T_num[-1]:.4f} °C (attendu {T_inf:.1f})")
print(f"Écart max au transitoire analytique : {np.max(np.abs(T_num - T_exact)):.3f} °C "
      f"(Euler implicite, ordre 1)")
assert abs(T_num[-1] - T_inf) < 1e-3, "la valeur finale doit être exacte"
print("OK")