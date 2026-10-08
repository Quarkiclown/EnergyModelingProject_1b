import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.components.sources.boundary import Inlet
from caldis.components.HEX.heat_exchanger import NTUHeatExchanger
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.steady import solve_steady

be_r = CoolPropBackend("R32")
be_a = CoolPropBackend("Air")
p_a, T_a_in, mdot_a = 25e5, 353.15, 0.02    # R32 surchauffe 80 °C
p_b, T_b_in, mdot_b = 1e5, 293.15, 0.2     # air 20 °C
h_a_in = PropsSI("H", "P", p_a, "T", T_a_in, "R32")
h_b_in = PropsSI("H", "P", p_b, "T", T_b_in, "Air")

def build():
    s = System("hx")
    ina = Inlet("ina", p=p_a, h=h_a_in, mdot=mdot_a)
    inb = Inlet("inb", p=p_b, h=h_b_in, mdot=mdot_b)
    hx = NTUHeatExchanger("hx", be_r, be_a, UA=10.0)
    s.add(ina, inb, hx)
    s.connect(ina.out, hx.a_in); s.connect(inb.out, hx.b_in)
    s.assemble()
    for port in (ina.out, hx.a_in, hx.a_out):
        port.p.start, port.h.start = p_a, h_a_in
    for port in (inb.out, hx.b_in, hx.b_out):
        port.p.start, port.h.start = p_b, h_b_in
    hx.a_in.mdot.start, hx.a_out.mdot.start = mdot_a, -mdot_a
    hx.b_in.mdot.start, hx.b_out.mdot.start = mdot_b, -mdot_b
    return s, hx

UA_list = np.linspace(1.0, 600.0, 100)
# (Ta_out, Tb_out, Q) par arrangement
res = {"counterflow": ([], [], []), "parallelflow": ([], [], [])}
for arr in res:
    s, hx = build()
    hx.params["arrangement"] = arr
    for UA in UA_list:
        hx.params["UA"] = UA
        sol = solve_steady(s); assert sol.success, f"{arr} UA={UA}: {sol.message}"
        Ta = be_r.T(hx.a_out.p.value, hx.a_out.h.value) - 273.15
        Tb = be_a.T(hx.b_out.p.value, hx.b_out.h.value) - 273.15
        Q = mdot_a * (h_a_in - hx.a_out.h.value) / 1e3
        res[arr][0].append(Ta); res[arr][1].append(Tb); res[arr][2].append(Q)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# gauche : temperatures de sortie des DEUX fluides
ax[0].plot(UA_list, res["counterflow"][0], '-',  c='r', label="R32 sortie (contre)")
ax[0].plot(UA_list, res["parallelflow"][0], '--', c='r', label="R32 sortie (co)")
ax[0].plot(UA_list, res["counterflow"][1], '-',  c='b', label="air sortie (contre)")
ax[0].plot(UA_list, res["parallelflow"][1], '--', c='b', label="air sortie (co)")
ax[0].axhline(T_a_in - 273.15, ls=':', c='r', lw=0.8, label="R32 entrée 80°C")
ax[0].axhline(T_b_in - 273.15, ls=':', c='b', lw=0.8, label="air entrée 20°C")
ax[0].set_xlabel("UA (W/K)"); ax[0].set_ylabel("T sortie (°C)")
ax[0].legend(fontsize=8); ax[0].grid(True)
# droite : puissance echangee
ax[1].plot(UA_list, res["counterflow"][2], '-',  label="contre-courant")
ax[1].plot(UA_list, res["parallelflow"][2], '--', label="co-courant")
ax[1].set_xlabel("UA (W/K)"); ax[1].set_ylabel("Q (kW)"); ax[1].legend(); ax[1].grid(True)
fig.suptitle("e-NTU R32/air — contre-courant vs co-courant")
fig.tight_layout(); plt.savefig("test_ntu_hx.png", dpi=110); plt.show()

print(f"UA=60 : Q contre = {res['counterflow'][2][-1]:.3f} kW, "
      f"co = {res['parallelflow'][2][-1]:.3f} kW "
      f"(contre >= co attendu : {res['counterflow'][2][-1] >= res['parallelflow'][2][-1]})")