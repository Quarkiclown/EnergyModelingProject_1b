"""Objet Results : accès hiérarchique par attributs à l'historique d'une simulation.

Construit un arbre à partir des clés plates ("tank.fluid.M", "tank.wall.T", ...).
  - res.time                    -> axe (temps, ou paramètre balayé)
  - res.tank.wall.T             -> série (np.ndarray) de longueur N
  - res.tank.fluid["in"].mdot   -> repli crochets ('in' est un mot-clé Python)
  - res["tank.fluid.in.p"]      -> accès plat par clé complète
  - res.keys() / "clé" in res   -> introspection
  - res.to_dataframe()          -> pandas (colonnes = grandeurs, + 'time')

Feuille = np.ndarray ; nœud interne = navigable. Rien n'est recalculé ici : Results ne
fait que présenter ce que collect_outputs a déjà enregistré pas à pas.
"""

import numpy as np


class _Node:
    def __init__(self, name):
        self._name = name
        self._children = {}
        self._series = None

    def _add(self, segments, series):
        if not segments:
            self._series = series
            return
        head, *rest = segments
        child = self._children.get(head)
        if child is None:
            child = _Node(f"{self._name}.{head}" if self._name else head)
            self._children[head] = child
        child._add(rest, series)

    def _resolve(self):
        if self._series is not None and not self._children:
            return self._series          # feuille pure -> le tableau
        return self                      # nœud navigable

    def __getitem__(self, key):
        if key in self._children:
            return self._children[key]._resolve()
        raise KeyError(f"{self._name or '<racine>'}: pas de sous-élément {key!r} "
                       f"(disponibles : {sorted(self._children)})")

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    @property
    def value(self):
        if self._series is None:
            raise AttributeError(f"{self._name!r} n'est pas une feuille")
        return self._series

    def __dir__(self):
        return list(self._children.keys())

    def __repr__(self):
        if self._series is not None and not self._children:
            return f"<série {self._name!r} n={len(self._series)}>"
        return f"<nœud {self._name!r} : {sorted(self._children)}>"


class Results:
    def __init__(self, time, hist):
        self.time = np.asarray(time)
        self._hist = {k: np.asarray(v) for k, v in hist.items()}
        self._root = _Node("")
        for key, series in self._hist.items():
            self._root._add(key.split("."), series)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._root[key]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __getitem__(self, key):
        if key in self._hist:
            return self._hist[key]
        node = self._root
        for seg in key.split("."):
            if not isinstance(node, _Node):
                raise KeyError(f"{key!r}: segment de trop après une feuille")
            node = node[seg]
        return node

    def __contains__(self, key):
        return key in self._hist

    def __iter__(self):
        # rétrocompat : t, hist = simulate(...)
        yield self.time
        yield self._hist

    def keys(self):
        return sorted(self._hist)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({"time": self.time, **self._hist})

    def __dir__(self):
        tops = {k.split(".")[0] for k in self._hist}
        return list(tops) + ["time", "keys", "to_dataframe"]

    def __repr__(self):
        tops = sorted({k.split(".")[0] for k in self._hist})
        return f"<Results N={len(self.time)} : {tops}>"