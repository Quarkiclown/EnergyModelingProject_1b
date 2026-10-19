import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.core.dof import check_dof
from caldis.components.sources.boundary import Inlet
from caldis.components.HEX.heat_exchanger import EffectivenessHX
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.steady import solve_steady

be_r = CoolPropBackend("R32")   # circuit A : réfrigérant chaud
be_a = CoolPropBackend("Air")   # circuit B : air froid
p_a, T_a_in, mdot_a = 25e5, 343.15, 0.0002   # R32 vapeur surchauffée, 70 °C
p_b, T_b_in, mdot_b = 1e5, 298.15, 0.05    # air, 25 °C
h_a_in = PropsSI("H", "P", p_a, "T", T_a_in, "R32")
h_b_in = PropsSI("H", "P", p_b, "T", T_b_in, "Air")

sys_ = System("hx")
ina = Inlet("ina", p=p_a, h=h_a_in, mdot=mdot_a)
inb = Inlet("inb", p=p_b, h=h_b_in, mdot=mdot_b)
hx = EffectivenessHX("hx", backend_a=be_r, backend_b=be_a, eff=0.5)
sys_.add(ina, inb, hx)
sys_.connect(ina.out, hx.a_in)
sys_.connect(inb.out, hx.b_in)
sys_.assemble()
for port in (ina.out, hx.a_in, hx.a_out):
    port.p.start, port.h.start = p_a, h_a_in
for port in (inb.out, hx.b_in, hx.b_out):
    port.p.start, port.h.start = p_b, h_b_in
hx.a_in.mdot.start, hx.a_out.mdot.start = mdot_a, -mdot_a
hx.b_in.mdot.start, hx.b_out.mdot.start = mdot_b, -mdot_b
print(check_dof(sys_))

effs = np.linspace(0.9, 0.99, 10)
Ta_out, Tb_out, Q_a, Q_b = [], [], [], []
for e in effs:
    hx.params["eff"] = e
    sol = solve_steady(sys_); assert sol.success, sol.message
    Ta_out.append(be_r.T(hx.a_out.p.value, hx.a_out.h.value) - 273.15)
    Tb_out.append(be_a.T(hx.b_out.p.value, hx.b_out.h.value) - 273.15)
    Q_a.append(mdot_a * (h_a_in - hx.a_out.h.value) / 1e3)   # kW côté R32
    Q_b.append(mdot_b * (hx.b_out.h.value - h_b_in) / 1e3)   # kW côté air

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].axhline(T_a_in - 273.15, ls=':', c='r', label="R32 entrée 70°C")
ax[0].axhline(T_b_in - 273.15, ls=':', c='b', label="air entrée 25°C")
ax[0].plot(effs, Ta_out, 'o-', c='r', label="R32 sortie")
ax[0].plot(effs, Tb_out, 's-', c='b', label="air sortie")
ax[0].set_xlabel("efficacité"); ax[0].set_ylabel("T (°C)"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(effs, Q_a, 'o-', label="côté R32")
ax[1].plot(effs, Q_b, 'x--', label="côté air")
ax[1].set_xlabel("efficacité"); ax[1].set_ylabel("puissance échangée (kW)")
ax[1].legend(); ax[1].grid(True)
fig.suptitle("Échangeur à efficacité — R32 chaud / air froid")
fig.tight_layout(); plt.savefig("test_heat_exchanger.png", dpi=110); plt.show()
print(f"écart max |Q_R32 - Q_air| = {max(abs(a-b) for a,b in zip(Q_a,Q_b)):.4f} kW (doit être ~0)")