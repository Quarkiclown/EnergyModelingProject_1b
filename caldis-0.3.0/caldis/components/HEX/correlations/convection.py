"""WP2 -- convective heat-transfer coefficient h_conv [W/m2/K]. brick(ctx) -> h."""


def h_conv_constant(h=500.0):
    """Simplest default: a constant coefficient, value chosen at simulation time."""
    def brick(ctx):
        return h
    brick.__name__ = "h_conv_constant"
    return brick


def h_conv_dittus_boelter(C=0.023, m=0.8, n=0.4, Nu_lam=4.0, Re_turb=2300.0):
    """First realistic correlation: Nu = C*Re^m*Pr^n (turbulent), with a laminar
    floor Nu = Nu_lam below Re_turb. h = Nu * k / dh."""
    def brick(ctx):
        Re = max(ctx.Re, 1e-9)
        Pr = max(ctx.Pr, 1e-9)
        Nu = Nu_lam if Re < Re_turb else C * Re ** m * Pr ** n
        dh = ctx.dh if ctx.dh > 1e-12 else 1e-3
        return Nu * ctx.k / dh
    brick.__name__ = "h_conv_dittus_boelter"
    return brick
