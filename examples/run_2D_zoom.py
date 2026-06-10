from idefix2python import RunContext, Pipeline, Fig, MapMovie2D
import numpy as np
from pathlib import Path

projectPath = Path(__file__).parent / "data_examples"
task = "2D_test"
configPath = projectPath / "config.json"


# If we want to show only the z>0 part of the disk
def zoom(x1, x2):
    # In this run geometry is spherical so x1, x2 correspond to r, theta
    return np.ones_like(x1, dtype=bool), x2 <= np.pi / 2


def compute_mach_p(v):
    data = v.data
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
fig0 = Fig(quantities, suptitle="Density and Mach number on a beautiful heatmap")

runContext = RunContext(
    task, projectPath, configPath=configPath, zoom=zoom, custom_name="2D_test_zoom"
)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
