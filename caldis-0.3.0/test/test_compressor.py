import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.components.sources.boundary import Inlet, Sink
from caldis.components.machines.compressor import Compressor
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.steady import solve_steady

FLUID = "R32"
be = CoolPropBackend(FLUID)
p_in = 8e5
h_in = PropsSI("H", "P", p_in, "T", 288.15, FLUID)   # 15 °C, surchauffé
mdot, eta = 0.02, 0.7

sys_ = System("comp")
inlet = Inlet("in", p=p_in, h=h_in, mdot=mdot)
comp = Compressor("comp", backend=be, eta_is=eta)
snk = Sink("snk", p=20e5)
sys_.add(inlet, comp, snk)
sys_.connect(inlet.out, comp.inl)
sys_.connect(comp.out, snk.inl)
sys_.assemble()
for port in (inlet.out, comp.inl):
    port.p.start, port.h.start = p_in, h_in
for port in (comp.out, snk.inl):
    port.p.start, port.h.start = 20e5, h_in + 3e4

p_outs = np.linspace(12e5, 40e5, 15)
T_out, T_out_is, W = [], [], []
for po in p_outs:
    snk.params["p"] = po
    sol = solve_steady(sys_); assert sol.success, sol.message
    h_out = comp.out.h.value
    T_out.append(be.T(po, h_out) - 273.15)
    T_out_is.append(be.T(po, be.h_ps(po, be.s(p_in, h_in))) - 273.15)
    W.append(mdot * (h_out - h_in) / 1e3)

pr = p_outs / p_in
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(pr, T_out, 'o-', label="réel (eta=0.7)")
ax[0].plot(pr, T_out_is, 's--', label="isentropique")
ax[0].set_xlabel("taux p_out/p_in"); ax[0].set_ylabel("T refoulement (°C)")
ax[0].legend(); ax[0].grid(True)
ax[1].plot(pr, W, 'o-')
ax[1].set_xlabel("taux p_out/p_in"); ax[1].set_ylabel("puissance absorbée (kW)"); ax[1].grid(True)
fig.suptitle(f"Compresseur {FLUID} — mdot={mdot} kg/s, eta_is={eta}")
fig.tight_layout(); plt.savefig("test_compressor.png", dpi=110); plt.show()