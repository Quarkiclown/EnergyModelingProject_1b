class FluidState:
    """État fluide construit depuis (p, h). T, rho, x dérivés via le backend."""

    def __init__(self, backend, p, h):
        self.backend = backend
        self.p = p
        self.h = h

    @property
    def T(self):
        return self.backend.T(self.p, self.h)

    @property
    def rho(self):
        return self.backend.rho(self.p, self.h)

    @property
    def x(self):
        return self.backend.quality(self.p, self.h)

    def __repr__(self):
        return f"FluidState(p={self.p:.4g}, h={self.h:.4g}, T={self.T:.4g})"