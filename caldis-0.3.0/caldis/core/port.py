from caldis.core.variable import Variable

class SignalPort:

    """Port de signal (stub)."""

    def __init__(self, owner, name, start=0.0):
        self.owner = owner
        self.name = name
        self.value = Variable(owner, name, start=start)

    @property
    def path(self):
        return f"{self.owner.path}.{self.name}"

    def variables(self):

        return [self.value]



class FluidPort:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self._p = Variable(self, "p", start=1e5, lower=0.0)
        self._mdot = Variable(self, "mdot", start=0.01)
        self._h = Variable(self, "h", start=3e5)
        self.backend = None

    def exergy_flow(self, T0, p0):
        """Flow exergy rate carried by this port [W], relative to the dead state (T0, p0):
            Xdot = |mdot| * [ (h - h0) - T0 * (s - s0) ]
        Uses the magnitude of mdot so that, for any component, the exergy DROP across a
        stream reads as inlet.exergy_flow - outlet.exergy_flow. Requires a backend."""
        be = self.backend
        if be is None:
            raise AttributeError(
                f"{self.path}: no fluid attached -> cannot compute exergy_flow")
        h0 = be.h_pt(p0, T0)
        s0 = be.s(p0, h0)
        s = be.s(self.p, self.h)
        psi = (self.h - h0) - T0 * (s - s0)
        return abs(self.mdot) * psi

    @property
    def p(self): return self._p.value
    @property
    def mdot(self): return self._mdot.value
    @property
    def h(self): return self._h.value

    # thermodynamic properties (unchanged, already values)
    @property
    def T(self):
        if self.backend is None:
            raise AttributeError(f"{self.path}: no fluid attached, cannot compute T")
        return self.backend.T(self._p.value, self._h.value)
    @property
    def T_C(self): return self.T - 273.15
    @property
    def x(self):
        if self.backend is None:
            raise AttributeError(f"{self.path}: no fluid attached, cannot compute x")
        return self.backend.quality(self._p.value, self._h.value)
    @property
    def s(self):
        if self.backend is None:
            raise AttributeError(f"{self.path}: no fluid attached, cannot compute s")
        return self.backend.s(self._p.value, self._h.value)

    @property
    def path(self):
        return f"{self.owner.path}.{self.name}"

    def variables(self):
        return [self._p, self._mdot, self._h]