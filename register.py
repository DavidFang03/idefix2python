"""
This file contains some nomenclature.
For example, the name of every possible field depending on the geometry:
VX1 should be $v_r$ or $v_x$ depending on the geometry.
"""


def alias(geometry, key):
    # https://idefix.readthedocs.io/latest/reference/definitions.hpp.html
    if geometry == "spherical":
        aliases = {"1": "R", "2": r"\theta", "3": r"\phi"}
    elif geometry == "cylindrical":
        aliases = {"1": "r", "2": r"z", "3": r"\phi"}
    elif geometry == "polar":
        aliases = {"1": "r", "2": r"\phi", "3": r"z"}
    elif geometry == "cartesian":
        aliases = {"1": "x", "2": r"y", "3": r"z"}
    else:
        raise NotImplementedError(f"What is {geometry}??")

    component = key[-1]
    firstletter_fieldnature = key[-3:-1]
    # BX or VX
    if firstletter_fieldnature in ["BX", "VX"]:
        nature = firstletter_fieldnature[0]
        alias = rf"${nature}_{aliases[component]}$"
    # RHO
    else:
        if "RHO" in key:
            alias = r"\rho"
        elif "cs" in key:
            alias = r"c_\text{s}"

        # my custom stuff
        elif "Mach_p" in key:
            alias = r"$\text{Mach}_\text{p}$"
        elif "eta" in key:
            alias = r"$\eta_\text{O}"
        elif "Am" in key:
            alias = r"$\text{Am}"

    if "Dust" in key:
        alias += r"^\text{dust}"

    return alias
