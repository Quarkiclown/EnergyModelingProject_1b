import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.components.sources.boundary import Source, Sink
from caldis.components.valves.expansion_valve import ExpansionValve
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.steady import solve_steady

FLUID = "R32"
be = CoolPropBackend(FLUID)
p_in = 25e5
h_in = PropsSI("H", "P", p_in, "T", 303.15, FLUID)   # 30 °C, liquide sous-refroidi
p_out, K = 8e5, 1e-8

sys_ = System("valve")
src = Source("src", p=p_in, h=h_in)
vlv = ExpansionValve("vlv", K=K, opening=1.0)
snk = Sink("snk", p=p_out)
sys_.add(src, vlv, snk)
sys_.connect(src.out, vlv.inl)
sys_.connect(vlv.out, snk.inl)
sys_.assemble()
for port in (src.out, vlv.inl, vlv.out, snk.inl):
    port.p.start, port.h.start = p_in, h_in

ops = np.linspace(0.05, 1.0, 20)
mdot_op, x_op = [], []
for op in ops:
    vlv.params["opening"] = op
    sol = solve_steady(sys_); assert sol.success, sol.message
    mdot_op.append(vlv.inl.mdot.value * 1e3)
    x_op.append(be.quality(p_out, vlv.out.h.value))

vlv.params["opening"] = 0.5
dps = np.linspace(2e5, 20e5, 20)
mdot_dp = []
for dp in dps:
    snk.params["p"] = p_in - dp
    sol = solve_steady(sys_); assert sol.success, sol.message
    mdot_dp.append(vlv.inl.mdot.value * 1e3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ops, mdot_op, 'o-')
ax[0].set_xlabel("ouverture [0-1]"); ax[0].set_ylabel("débit (g/s)"); ax[0].grid(True)
ax[0].set_title(f"dp={(p_in-p_out)/1e5:.0f} bar — linéaire en ouverture")
ax[1].plot(dps / 1e5, mdot_dp, 's-')
ax[1].set_xlabel("dp (bar)"); ax[1].set_ylabel("débit (g/s)"); ax[1].grid(True)
ax[1].set_title("ouverture=0.5 — linéaire en dp")
fig.suptitle(f"Détendeur linéaire {FLUID} — x sortie ~ {np.mean(x_op):.3f} (constant, isenthalpique)")
fig.tight_layout(); plt.savefig("test_expansion_valve.png", dpi=110); plt.show()
print(f"x sortie (constant) ~ {np.mean(x_op):.4f}")
