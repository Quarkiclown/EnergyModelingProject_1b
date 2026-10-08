import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.components.sources.boundary import FlowSource, Sink
from caldis.components.valves.valve import Valve
from caldis.components.vessels.control_volume import ControlVolume
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.dynamic import simulate
from caldis.solvers.steady import solve_steady

FLUID = "R290"
be = CoolPropBackend(FLUID)

p0 = 5.0e5
h_liq = PropsSI("H", "P", p0, "Q", 0.0, FLUID)
mdot = 0.01
Q0, Q1 = 500.0, 1000.0
h_in = h_liq - 20_000.0
h0 = h_in + Q0 / mdot

p_sink = 4.0e5
Kv = mdot / np.sqrt(p0 - p_sink)

sys_ = System("evaporateur")
src = FlowSource("src", mdot=mdot, h=h_in)
evap = ControlVolume("evap", backend=be, V=0.01, p_init=p0, h_init=h0, Qdot=Q0)
vlv = Valve("vlv", Kv=Kv)
snk = Sink("snk", p=p_sink)

sys_.add(src, evap, vlv, snk)
sys_.connect(src.out, evap.inl)
sys_.connect(evap.out, vlv.inl)
sys_.connect(vlv.out, snk.inl)

def update_inputs(system, t):
    evap.params["Qdot"] = Q1 if t > 0.0 else Q0

# fenetre longue : constante de temps ~ M/mdot ~ 100 s -> on va a ~15 tau
t, hist = simulate(sys_, t_end=1500.0, dt=2.0, update_inputs=update_inputs)

p = hist["evap.p"]; h = hist["evap.h"]
print(f"{FLUID} — echelon Q {Q0:.0f} -> {Q1:.0f} W\n")
print(f"{'t':>5s} {'p(bar)':>9s} {'h(kJ/kg)':>10s} {'T(°C)':>8s} {'x':>8s} {'mdot_out':>10s}")
for i in range(0, len(t), len(t) // 10):
    T = be.T(p[i], h[i]); x = be.quality(p[i], h[i])
    print(f"{t[i]:5.0f} {p[i]/1e5:9.4f} {h[i]/1e3:10.3f} {T-273.15:8.3f} {x:8.4f} "
          f"{hist['evap.out.mdot'][i]:10.5f}")

h_expected = h_in + Q1 / mdot          # regime permanent : bilan d'energie

# validation 1 : point final dynamique vs analytique
err_h = abs(h[-1] - h_expected)
err_m = abs(hist["evap.out.mdot"][-1] + mdot)      # sortie = -entree a l'equilibre
print(f"\n[dynamique t={t[-1]:.0f}s] h = {h[-1]/1e3:.4f} kJ/kg (attendu {h_expected/1e3:.4f})")
print(f"   ecart h = {err_h:.3f} J/kg ; desequilibre masse = {err_m:.2e} kg/s")

# validation 2 : permanent independant (autre point de depart), doit retomber au meme etat
sol_ss = solve_steady(sys_)            # repart du guess de construction (etat Q0)
h_ss = evap.h.value
print(f"[permanent solve_steady] convergence={sol_ss.success} ; h = {h_ss/1e3:.4f} kJ/kg")

assert err_h < 1e-3 * abs(h_expected), "point final dynamique != regime attendu"
assert err_m < 1e-3 * mdot, "bilan de masse non ferme a l'equilibre"
assert sol_ss.success and abs(h_ss - h_expected) < 1e-4 * abs(h_expected), \
    "le permanent independant ne retrouve pas le meme etat"
print("\nOK : dynamique et permanent convergent vers le meme etat, conservation verifiee.")