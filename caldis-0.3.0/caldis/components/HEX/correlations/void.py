"""WP5 -- void fraction eps [-]. brick(ctx) -> eps in [0, 1].

Homogeneous (no slip) and Zivi (slip = (rho_l/rho_g)^(1/3)) share the SAME form,
only the density-ratio exponent differs (1 vs 2/3) -- a clean way to see the effect
of phase slip. Single-phase returns 0 (liquid) or 1 (vapour).
"""


def _void_slip(ctx, exponent):
    x = ctx.x
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if ctx.rho_l <= 0.0 or ctx.rho_g <= 0.0:
        return x
    return 1.0 / (1.0 + ((1.0 - x) / x) * (ctx.rho_g / ctx.rho_l) ** exponent)


def void_homogeneous():
    """Simplest default: homogeneous model, eps = 1/(1 + ((1-x)/x)*(rho_g/rho_l))."""
    def brick(ctx):
        return _void_slip(ctx, 1.0)
    brick.__name__ = "void_homogeneous"
    return brick


def void_zivi():
    """First realistic model: Zivi slip, exponent 2/3 instead of 1."""
    def brick(ctx):
        return _void_slip(ctx, 2.0 / 3.0)
    brick.__name__ = "void_zivi"
    return brick
