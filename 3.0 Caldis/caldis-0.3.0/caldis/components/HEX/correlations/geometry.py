"""WP1 -- geometry. geometry_fn(element_index, geometry_params) -> dict.

Not a brick(ctx): geometry is resolved first (it is void- and state-free) and feeds
the per-element context. One variant for now (simplified smooth plate channel).
"""


def geometry_plate(element_index, geometry_params):
    """Simplified plate channel, identical for every element.
    geometry_params: plate_width [m], plate_gap [m], L_total [m], n_elements [-]."""
    w = geometry_params.get("plate_width", 0.1)
    b = geometry_params.get("plate_gap", 0.003)
    L_total = geometry_params.get("L_total", 0.5)
    n = geometry_params.get("n_elements", 10)

    L_elem = L_total / n
    A_cross = w * b
    A_exchange = w * L_elem
    V_internal = A_cross * L_elem
    perimeter_wet = 2.0 * (w + b)
    dh = 4.0 * A_cross / perimeter_wet

    return {
        "A_exchange": A_exchange, "A_cross": A_cross, "V_internal": V_internal,
        "dh": dh, "perimeter_wet": perimeter_wet, "L_elem": L_elem,
        "factor_geometry": 1.0,
        "fin_length": 0.0, "fin_thickness": 0.0, "fin_perimeter": 0.0, "fin_k": 0.0,
    }
