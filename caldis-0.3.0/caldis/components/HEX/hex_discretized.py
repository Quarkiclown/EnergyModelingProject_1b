"""Discretised heat-exchanger model in SIMULATION mode, for Caldis 0.2.0.

Residual-based finite-volume (cell-centred), co- or counter-flow. Per fluid cell:
states M, U (differential); algebraic p, h. Internal face flows closed by momentum.
One wall temperature per element. Full dynamic form; the solver sets the regime
(solve -> steady der=0, simulate -> transient).

Physics enters through replaceable BRICKS, selected with their parameters at
simulation time (see correlations/). Each brick is brick(ctx) -> value. The skeleton
owns the per-cell context (two-stage build: void-free then void-dependent), exposes
the wall temperature (ctx.T_wall) and the actual face flow (ctx.mdot_face) so a T^4
radiation model and a quadratic pressure drop are exact, and never freezes a coupled
quantity.
"""

from caldis.core.component import Component
from caldis.components.HEX.element_context import ElementContext
from caldis.components.HEX.correlations.convection import h_conv_constant
from caldis.components.HEX.correlations.radiation import h_rad_constant
from caldis.components.HEX.correlations.pressure import dp_linear
from caldis.components.HEX.correlations.void import void_homogeneous
from caldis.components.HEX.correlations.geometry import geometry_plate


class HXSim(Component):

    def __init__(self, name, backend_hot, backend_cold,
                 n_elements=10, flow_config="counterflow", geometry_params=None,
                 m_wall=1.0, cp_wall=500.0, R_fouling=0.0, geometry_fn=None,
                 h_conv_hot=None, h_conv_cold=None,
                 h_rad_hot=None,  h_rad_cold=None,
                 dp_hot=None,     dp_cold=None,
                 void_hot=None,   void_cold=None):
        
        super().__init__(name)
        if flow_config not in ("counterflow", "coflow"):
            raise ValueError(f"flow_config must be 'counterflow' or 'coflow', got {flow_config!r}")
        self.backend_hot = backend_hot
        self.backend_cold = backend_cold
        self.n = n_elements
        self.flow_config = flow_config
        self.geometry_params = dict(geometry_params or {})
        self.geometry_params.setdefault("n_elements", n_elements)
        self.params.update(m_wall=m_wall, cp_wall=cp_wall, R_fouling=R_fouling)

        self.geometry_fn = geometry_fn or geometry_plate

        self.h_conv_hot  = h_conv_hot  or h_conv_constant()
        self.h_conv_cold = h_conv_cold or h_conv_constant()
        self.h_rad_hot   = h_rad_hot   or h_rad_constant(0.0)
        self.h_rad_cold  = h_rad_cold  or h_rad_constant(0.0)
        self.dp_hot      = dp_hot      or dp_linear()
        self.dp_cold     = dp_cold     or dp_linear()
        self.void_hot    = void_hot    or void_homogeneous()
        self.void_cold   = void_cold   or void_homogeneous()

        self.hot_in = self.add_port("hot_in", backend=backend_hot)
        self.hot_out = self.add_port("hot_out", backend=backend_hot)
        self.cold_in = self.add_port("cold_in", backend=backend_cold)
        self.cold_out = self.add_port("cold_out", backend=backend_cold)

        n = self.n
        # neutral, fluid-aware construction starts (overwritten by seed_internal_starts)
        p0h, h0h = backend_hot.default_state(283.15)
        p0c, h0c = backend_cold.default_state(283.15)
        V0 = self.geometry_fn(0, self.geometry_params)["V_internal"]
        rh = backend_hot.rho(p0h, h0h); Mh0 = rh * V0; Uh0 = V0 * (rh * h0h - p0h)
        rc = backend_cold.rho(p0c, h0c); Mc0 = rc * V0; Uc0 = V0 * (rc * h0c - p0c)
        Tw0 = 0.5 * (backend_hot.T(p0h, h0h) + backend_cold.T(p0c, h0c))
        m0 = 0.05

        self._Mh, self._Uh, self._ph, self._hh = [], [], [], []
        self._Mc, self._Uc, self._pc, self._hc = [], [], [], []
        self._Tw = []
        for i in range(n):
            self._Mh.append(self.add_variable(f"Mh_{i}", start=Mh0, differential=True, lower=0.0, scale=max(Mh0, 1e-6)))
            self._Uh.append(self.add_variable(f"Uh_{i}", start=Uh0, differential=True, scale=max(abs(Uh0), 1.0)))
            self._ph.append(self.add_variable(f"ph_{i}", start=p0h, lower=0.0, scale=1e5))
            self._hh.append(self.add_variable(f"hh_{i}", start=h0h, scale=1e5))
            self._Mc.append(self.add_variable(f"Mc_{i}", start=Mc0, differential=True, lower=0.0, scale=max(Mc0, 1e-6)))
            self._Uc.append(self.add_variable(f"Uc_{i}", start=Uc0, differential=True, scale=max(abs(Uc0), 1.0)))
            self._pc.append(self.add_variable(f"pc_{i}", start=p0c, lower=0.0, scale=1e5))
            self._hc.append(self.add_variable(f"hc_{i}", start=h0c, scale=1e5))
            self._Tw.append(self.add_variable(f"Tw_{i}", start=Tw0, differential=True, scale=1e2))
        self._mdot_fh = [self.add_variable(f"mdot_fh_{k}", start=m0, scale=max(m0, 1e-4)) for k in range(n - 1)]
        self._mdot_fc = [self.add_variable(f"mdot_fc_{k}", start=m0, scale=max(m0, 1e-4)) for k in range(n - 1)]

    def _order(self, side):
        n = self.n
        if side == "hot":
            return list(range(n))
        if self.flow_config == "counterflow":
            return list(range(n - 1, -1, -1))
        return list(range(n))

    def _ctx(self, side, i, p, h, mdot_nom, T_wall, be):
        c = ElementContext(side=side, index=i, p=p, h=h, mdot=abs(mdot_nom),
                           T_wall=T_wall, R_fouling=self.params["R_fouling"],
                           params=self.params)
        # stage 1: thermodynamics + transport (void-free)
        c.T = be.T(p, h)
        c.x = be.quality(p, h)
        c.rho = be.rho(p, h)
        c.h_g = be.h_pq(p, 1.0); c.h_l = be.h_pq(p, 0.0)
        c.rho_g = be.rho(p, c.h_g); c.rho_l = be.rho(p, c.h_l)
        c.mu = be.mu(p, h); c.k = be.cond(p, h); c.cp = be.cp(p, h)
        c.Pr = be.Pr(p, h); c.sigma = be.sigma(p, h)
        g = self.geometry_fn(i, self.geometry_params)
        for key in ("A_exchange", "A_cross", "V_internal", "dh", "perimeter_wet",
                    "L_elem", "factor_geometry", "fin_length", "fin_thickness",
                    "fin_perimeter", "fin_k"):
            setattr(c, key, g[key])
        c.G = c.mdot / c.A_cross if c.A_cross > 1e-30 else 0.0
        if 0.0 < c.x < 1.0 and c.rho_g > 0 and c.rho_l > 0:
            c.j_g = c.G * c.x / c.rho_g
            c.j_l = c.G * (1.0 - c.x) / c.rho_l
        else:
            c.j_g = c.G / c.rho if (c.x >= 1.0 and c.rho > 0) else 0.0
            c.j_l = c.G / c.rho if (c.x <= 0.0 and c.rho > 0) else 0.0
        # stage 2: void -> hold-up density/energy, in-situ velocity, Re
        void_fn = self.void_hot if side == "hot" else self.void_cold
        c.eps = void_fn(c)

        if 0.0 < c.x < 1.0:
            c.rho_eff = c.eps * c.rho_g + (1.0 - c.eps) * c.rho_l
            c.rho_h_eff = c.eps * c.rho_g * c.h_g + (1.0 - c.eps) * c.rho_l * c.h_l
        else:
            c.rho_eff = c.rho
            c.rho_h_eff = c.rho * c.h
        c.v = c.mdot / (c.rho_eff * c.A_cross) if (c.rho_eff > 0 and c.A_cross > 0) else 0.0
        c.Re = c.G * c.dh / c.mu if (c.mu and c.mu == c.mu and abs(c.mu) > 1e-30) else 0.0
        return c

    def residuals(self):
        n = self.n
        be_h, be_c = self.backend_hot, self.backend_cold
        mdot_h_nom = abs(self.hot_in.mdot)
        mdot_c_nom = abs(self.cold_in.mdot)

        ctx_h = [self._ctx("hot", i, self._ph[i].value, self._hh[i].value,
                           mdot_h_nom, self._Tw[i].value, be_h) for i in range(n)]
        ctx_c = [self._ctx("cold", i, self._pc[i].value, self._hc[i].value,
                           mdot_c_nom, self._Tw[i].value, be_c) for i in range(n)]

        Qh = [0.0] * n
        Qc = [0.0] * n
        wall_res = []
        Rf = self.params["R_fouling"]
        m_wall_i = self.params["m_wall"] / n
        cp_wall = self.params["cp_wall"]
        for i in range(n):
            hh = self.h_conv_hot(ctx_h[i])  + self.h_rad_hot(ctx_h[i])
            hc = self.h_conv_cold(ctx_c[i]) + self.h_rad_cold(ctx_c[i])

            Uh = 1.0 / (1.0 / hh + Rf) if hh > 0 else 0.0
            Uc = 1.0 / (1.0 / hc + Rf) if hc > 0 else 0.0
            Th, Tc, Tw = ctx_h[i].T, ctx_c[i].T, self._Tw[i].value
            Ah, Ac = ctx_h[i].A_exchange, ctx_c[i].A_exchange
            q_h2w = Uh * Ah * (Th - Tw)
            q_c2w = Uc * Ac * (Tc - Tw)
            Qh[i] = -q_h2w
            Qc[i] = -q_c2w
            wall_res.append(m_wall_i * cp_wall * self._Tw[i].der - (q_h2w + q_c2w))

        res = []
        res += self._side_residuals("hot", self._Mh, self._Uh, self._ph, self._hh,
                                    self._mdot_fh, self.hot_in, self.hot_out, ctx_h, Qh)
        res += self._side_residuals("cold", self._Mc, self._Uc, self._pc, self._hc,
                                    self._mdot_fc, self.cold_in, self.cold_out, ctx_c, Qc)
        res += wall_res
        return res

    def _side_residuals(self, side, M, U, p, h, mdot_face,
                        inlet_port, outlet_port, ctx, Q):
        n = self.n
        order = self._order(side)
        r = []

        for k in range(n):
            i = order[k]
            if k == 0:
                mdot_in, h_in = inlet_port.mdot, inlet_port.h
            else:
                mdot_in = mdot_face[k - 1].value
                # upwind conscient du signe : si le flux s'inverse, l'amont change de côté
                h_in = h[order[k - 1]].value if mdot_in >= 0.0 else h[order[k]].value
            if k == n - 1:
                mdot_out = -outlet_port.mdot
            else:
                mdot_out = mdot_face[k].value
            h_out = h[i].value
            r.append(M[i].der - (mdot_in - mdot_out))
            r.append(U[i].der - (mdot_in * h_in - mdot_out * h_out + Q[i]))


        # EOS per cell (hold-up uses rho_eff from the void fraction)
        for i in range(n):
            c = ctx[i]
            r.append(M[i].value - c.rho_eff * c.V_internal)
            r.append(U[i].value - c.V_internal * (c.rho_h_eff - p[i].value))

        # momentum per internal face: dP uses the ACTUAL face flow (ctx.mdot_face)
        dp_fn = self.dp_hot if side == "hot" else self.dp_cold
        for k in range(n - 1):
            i0, i1 = order[k], order[k + 1]
            ctx[i0].mdot_face = mdot_face[k].value
            dP = dp_fn(ctx[i0]) * ctx[i0].L_elem
            r.append(p[i0].value - p[i1].value - dP)       
    

        # boundary couplings to the ports
        r.append(inlet_port.p - p[order[0]].value)
        r.append(outlet_port.p - p[order[n - 1]].value)
        r.append(outlet_port.h - h[order[n - 1]].value)
        return r

    def _own_start_sanity(self):
        checks = []
        def add(cond, msg):
            checks.append((bool(cond), msg))

        # --- inlet temperatures ---
        try:
            Th, Tc = self.hot_in.T, self.cold_in.T
            add(Th > Tc,
                f"{self.path}: hot inlet ({Th-273.15:.0f}°C) should be hotter than "
                f"cold inlet ({Tc-273.15:.0f}°C)")
        except Exception:
            pass  # no backend / state not evaluable at the start point: skip, no warning

        # --- port flow signs (convention: inlet > 0, outlet < 0) ---
        add(self.hot_in.mdot  > 0, f"{self.path}: hot_in flow should be > 0 (inflow)")
        add(self.cold_in.mdot > 0, f"{self.path}: cold_in flow should be > 0 (inflow)")
        add(self.hot_out.mdot  < 0, f"{self.path}: hot_out flow should be < 0 (outflow)")
        add(self.cold_out.mdot < 0, f"{self.path}: cold_out flow should be < 0 (outflow)")

        # --- mass conservation at the start point (|in| ~ |out|) ---
        add(abs(self.hot_in.mdot + self.hot_out.mdot)
            <= 1e-3 * max(1.0, abs(self.hot_in.mdot)),
            f"{self.path}: hot flows unbalanced at start "
            f"(in={self.hot_in.mdot:+.4g}, out={self.hot_out.mdot:+.4g})")
        add(abs(self.cold_in.mdot + self.cold_out.mdot)
            <= 1e-3 * max(1.0, abs(self.cold_in.mdot)),
            f"{self.path}: cold flows unbalanced at start "
            f"(in={self.cold_in.mdot:+.4g}, out={self.cold_out.mdot:+.4g})")

        # --- pressure-drop direction (pressure should fall from inlet to outlet) ---
        add(self.hot_in.p >= self.hot_out.p,
            f"{self.path}: hot pressure should drop from inlet to outlet "
            f"(in={self.hot_in.p/1e5:.3f} bar, out={self.hot_out.p/1e5:.3f} bar)")
        add(self.cold_in.p >= self.cold_out.p,
            f"{self.path}: cold pressure should drop from inlet to outlet "
            f"(in={self.cold_in.p/1e5:.3f} bar, out={self.cold_out.p/1e5:.3f} bar)")

        # --- face flows: no reversal at start (same sign as the inlet) ---
        if self.n >= 2:
            add(all(self._mdot_fh[k].value > 0 for k in range(self.n - 1)),
                f"{self.path}: hot face flows should be > 0 at start "
                f"(consistent with hot_in > 0)")
            add(all(self._mdot_fc[k].value > 0 for k in range(self.n - 1)),
                f"{self.path}: cold face flows should be > 0 at start "
                f"(consistent with cold_in > 0)")

        return checks
    
    def update_outputs(self):
        n = self.n
        be_h, be_c = self.backend_hot, self.backend_cold
        self._outputs = {
            "T_hot": [be_h.T(self._ph[i].value, self._hh[i].value) for i in range(n)],
            "T_cold": [be_c.T(self._pc[i].value, self._hc[i].value) for i in range(n)],
            "x_hot": [be_h.quality(self._ph[i].value, self._hh[i].value) for i in range(n)],
            "x_cold": [be_c.quality(self._pc[i].value, self._hc[i].value) for i in range(n)],
            "charge_hot": sum(self._Mh[i].value for i in range(n)),
            "charge_cold": sum(self._Mc[i].value for i in range(n)),
            "Qdot_total": self.hot_in.mdot * (self.hot_in.h - self.hot_out.h),
        }

    def seed_internal_starts(self):
        be_h, be_c = self.backend_hot, self.backend_cold
        p_h, h_h = self.hot_in._p.start, self.hot_in._h.start
        p_c, h_c = self.cold_in._p.start, self.cold_in._h.start
        m_h = abs(self.hot_in._mdot.start)
        m_c = abs(self.cold_in._mdot.start)
        V = self.geometry_fn(0, self.geometry_params)["V_internal"]
        rho_h, rho_c = be_h.rho(p_h, h_h), be_c.rho(p_c, h_c)
        Tw0 = 0.5 * (be_h.T(p_h, h_h) + be_c.T(p_c, h_c))
        for i in range(self.n):
            self._ph[i].start = p_h; self._hh[i].start = h_h
            self._Mh[i].start = rho_h * V; self._Uh[i].start = V * (rho_h * h_h - p_h)
            self._pc[i].start = p_c; self._hc[i].start = h_c
            self._Mc[i].start = rho_c * V; self._Uc[i].start = V * (rho_c * h_c - p_c)
            self._Tw[i].start = Tw0
        for k in range(self.n - 1):
            self._mdot_fh[k].start = m_h
            self._mdot_fc[k].start = m_c
        self._apply_internal_overrides()      # explicit inits win over propagation

    def set_wall_init(self, value):
        """Semantic shortcut: initial wall temperature, scalar (uniform) or list
        (per-element profile). A real initial condition when inertia is simulated."""
        return self.set_internal_start("Tw", value)
    
    @property
    def T_hot(self): return self._outputs["T_hot"]
    @property
    def T_cold(self): return self._outputs["T_cold"]
    @property
    def Tw(self):
        """Wall temperature per element, read live from the Tw state variables."""
        return [self._Tw[i].value for i in range(self.n)]
    @property
    def x_hot(self): return self._outputs["x_hot"]
    @property
    def x_cold(self): return self._outputs["x_cold"]
    @property
    def charge_hot(self): return self._outputs["charge_hot"]
    @property
    def charge_cold(self): return self._outputs["charge_cold"]
    @property
    def Qdot_total(self): return self._outputs["Qdot_total"]
