"""WP3 -- radiative heat-transfer coefficient h_rad [W/m2/K]. brick(ctx) -> h_rad.

The skeleton keeps the energy balance LINEAR in (T_fluid - T_wall): Q = (h_conv +
h_rad)*A*(T_f - T_w). To recover the physical T^4 law without changing the balance,
h_rad is returned as the EQUIVALENT radiative coefficient:

    h_rad = emissivity * sigma * F_view * (T_f^2 + T_w^2) * (T_f + T_w)

so that  h_rad*(T_f - T_w) = emissivity*sigma*F_view*(T_f^4 - T_w^4)  exactly.
This needs the wall temperature, available as ctx.T_wall.
"""

SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann constant [W/m2/K4]


def h_rad_constant(h=0.0):
    """Simplest default: a constant radiative coefficient (0 = radiation off)."""
    def brick(ctx):
        return h
    brick.__name__ = "h_rad_constant"
    return brick


def h_rad_view_factor(emissivity=0.8, F_view=1.0):
    """First realistic model: grey-body exchange, T^4 law expressed as an equivalent
    linear coefficient using the fluid and wall temperatures (ctx.T, ctx.T_wall)."""
    def brick(ctx):
        Tf, Tw = ctx.T, ctx.T_wall
        return emissivity * SIGMA_SB * F_view * (Tf * Tf + Tw * Tw) * (Tf + Tw)
    brick.__name__ = "h_rad_view_factor"
    return brick
