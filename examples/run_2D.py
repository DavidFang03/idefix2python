from idefix2python import RunContext, Pipeline, Fig, MapMovie2D
import numpy as np
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "2D_test"


def compute_mach_p(data):
    cs2 = data["PRS"] / data["RHO"]
    return np.sqrt(data["VX1"] ** 2 + data["VX2"] ** 2) / cs2


quantities = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
    ),
    # Create a computed field for Mach_p and draw a contour at Mach_p = 1
    MapMovie2D(
        "Mach_p",
        r"$\mathcal{M}_p$",
        plot_coords=[0, 1],
        title="Poloidal Mach Number",
        compute=compute_mach_p,
        contours=[1],
        contour_color="green",
    ),
]
fig0 = Fig(quantities, suptitle="Density and Mach number on a heatmap")
runContext = RunContext(task, projectPath)
pipeline = Pipeline(runContext, [fig0])
pipeline.run()
