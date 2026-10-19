class Connection:
    """Nœud à deux ports. close_loop=True omet le résidu de masse (redondant sur la
    connexion qui referme une boucle) : garde p et h, laisse tomber mdot."""

class Connection:
    def __init__(self, port_a, port_b, close_loop=False):
        self.a = port_a
        self.b = port_b
        self.close_loop = close_loop

    def residuals(self):
        r = [self.a.p - self.b.p,
             self.a.h - self.b.h]
        if not self.close_loop:
            r.append(self.a.mdot + self.b.mdot)
        return r
    
    def __repr__(self):
        tag = " [close_loop]" if self.close_loop else ""
        return f"Connection({self.a.path} <-> {self.b.path}){tag}"