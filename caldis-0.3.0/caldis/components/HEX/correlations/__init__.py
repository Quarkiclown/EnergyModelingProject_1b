"""Replaceable physics correlations for HXSim, one module per block.

Each correlation is a FACTORY: it takes its parameters and returns a brick
`brick(ctx) -> value` (a closure capturing the parameters). Choose correlation and
parameters together, at simulation time:

    from caldis.components.exchangers.correlations.convection import h_conv_dittus_boelter
    from caldis.components.exchangers.correlations.void import void_zivi
    hx = HXSim(..., h_conv_fn=h_conv_dittus_boelter(), void_fn=void_zivi())

Each block offers a SIMPLE default and a first REALISTIC correlation. Students add a
new one by writing another factory in the right module -- nothing else changes.
"""
