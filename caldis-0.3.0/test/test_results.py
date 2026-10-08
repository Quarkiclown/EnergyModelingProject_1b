import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from caldis.core.system import System
from caldis.components.sources.boundary import FlowSource, ClosedEnd
from caldis.components.vessels.tank import TankConstV
from caldis.fluids.coolprop_backend import CoolPropBackend
from caldis.solvers.dynamic import simulate

be = CoolPropBackend("Air")
p0, T0 = 1e5, 293.15
h0 = PropsSI("H", "P", p0, "T", T0, "Air")

sys_ = System("tank")
feed = FlowSource("feed", mdot=0.0, h=h0)
tank = TankConstV("tank", backend=be, V=0.05, p_init=p0, h_init=h0,
            C_wall=800., T_wall_init=T0, UA_wall=30., Q_heater=150.)
cap = ClosedEnd("cap")
sys_.add(feed, tank, cap)
sys_.connect(feed.out, tank.inl)
sys_.connect(tank.out, cap.inl)
sys_.assemble()

res = simulate(sys_, t_end=400., dt=2., update_inputs=None, init=None)

# --- acces par attributs (le but) ---
t  = res.time
Tw = res.tank.wall.T
Tf = res.tank.fluid.T
Q  = res.tank.Q

# --- repli crochets pour un segment mot-cle Python ('in') ---
mdot_in = res.tank.fluid["inl"].mdot        # res.tank.fluid.in.p -> SyntaxError
p = res.tank.fluid.inl.p 

# --- equivalences dotted / flat ---
assert np.array_equal(res.tank.wall.T, res["tank.wall.T"])
assert "tank.fluid.M" in res
assert t.shape == Tw.shape == Tf.shape == Q.shape

# --- retrocompat : Results se depaquette en (t, hist), MEME objet ---
res2 = simulate(sys_, t_end=400., dt=2., update_inputs=None, init=None)
t2, hist = res2               # depaquetage via __iter__
assert np.array_equal(hist["tank.wall.T"], res2.tank.wall.T)
assert np.array_equal(t2, res2.time)


print("repr        :", res)
print("noeud tank  :", res.tank)
print("cles (6)    :", res.keys()[:6], "...")
print("OK Results")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, Tw - 273.15, 'r-', label="paroi  (res.tank.wall.T)")
ax.plot(t, Tf - 273.15, 'b-', label="fluide (res.tank.fluid.T)")
ax.set_xlabel("t (s)"); ax.set_ylabel("T (°C)"); ax.legend(); ax.grid(True)
ax.set_title("Accès hiérarchique par attributs")
fig.tight_layout(); plt.savefig("test_results.png", dpi=110); plt.show()