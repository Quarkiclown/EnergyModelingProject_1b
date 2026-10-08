class Variable:
    """Inconnue. Le nom complet est CALCULÉ depuis owner.path (plus figé), pour que le
    chemin reflète la place dans l'arbre de composants même après rattachement."""

    def __init__(self, owner, local_name, start=1.0, lower=None, upper=None,
                 scale=None, differential=False):
        self.owner = owner            # Component ou FluidPort : expose .path
        self.local_name = local_name
        self.value = float(start)
        self.start = float(start)
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.index = None
        self.differential = differential   # True = variable d'état (a une dérivée)
        self.der = 0.0

    @property
    def name(self):
        return f"{self.owner.path}.{self.local_name}"

    def __repr__(self):
        return f"Variable({self.name}={self.value:.6g})"