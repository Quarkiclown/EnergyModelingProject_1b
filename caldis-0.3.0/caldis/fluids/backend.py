"""Interface abstraite des propriétés fluides.

Toute implémentation prend (p, h) en entrée — jamais (p, T) — pour rester univoque
sous le dôme de saturation (cf. discussion sur les états). Elle doit fournir les
valeurs ET, à terme, les dérivées partielles nécessaires à l'AD.

Implémentations prévues :
  - ConstantPropertiesBackend : placeholder du squelette (ci-contre), sans dépendance.
  - CoolPropBackend           : AbstractState + first_partial_deriv, emballé en
                                callback CasADi (INCRÉMENT SUIVANT).
  - TabulatedBackend          : TTSE / bicubique, lisse (C1) et rapide.
"""

from abc import ABC, abstractmethod


class FluidBackend(ABC):
    @abstractmethod
    def T(self, p, h):
        ...

    @abstractmethod
    def rho(self, p, h):
        ...

    @abstractmethod
    def quality(self, p, h):
        """Titre vapeur x ; hors zone diphasique, renvoie -1 (sous-refroidi) ou 2
        (surchauffé), convention CoolProp."""
        ...

    # TODO : dT_dp_h, dT_dh_p, ... (dérivées partielles pour l'AD à travers la boîte noire)
