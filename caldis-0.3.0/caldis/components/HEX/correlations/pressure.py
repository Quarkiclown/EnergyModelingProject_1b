"""WP4 -- pressure gradient along a channel, dp/dz [Pa/m]. brick(ctx) -> dp/dz.

A correlation returns a pressure GRADIENT (per metre), not a total drop. The skeleton
multiplies it by the element length (ctx.L_elem) to get the pressure difference between
two consecutive cell centres. This keeps the result mesh-independent: refining the
discretisation does not change the total pressure drop, which depends on geometry
(length, diameter) and not on the number of cells.

The brick reads the ACTUAL face mass flow (ctx.mdot_face), so the momentum equation
genuinely closes that flow. The gradient is an ODD function of mdot_face, so the
pressure always falls in the real flow direction (works for both flow senses and for
counter-current channels). A zero default is avoided on purpose: with dp/dz = 0 the
flow distribution is ill-posed in dynamic mode.
"""


def dp_linear(K=1000.0):
    """Simplest default that still closes the flow: pressure gradient linear in the
    mass flow, dp/dz = K * mdot_face  [K in (Pa/m) per (kg/s)]."""
    def brick(ctx):
        return K * ctx.mdot_face
    brick.__name__ = "dp_linear"
    return brick


def dp_quadratic(K=1.0e4):
    """First realistic form: pressure gradient quadratic in the mass flow (turbulent
    friction), sign-preserving: dp/dz = K * mdot_face * |mdot_face|
    [K in (Pa/m) per (kg/s)^2]."""
    def brick(ctx):
        m = ctx.mdot_face
        return K * m * abs(m)
    brick.__name__ = "dp_quadratic"
    return brick