from idefix2python import RunContext, Pipeline, Fig, PartQuantity, MapMovie2D
from pathlib import Path
import numpy as np

projectPath = Path(__file__).parent / "data_test"
task = "multipart_2D"
configPath = projectPath / "config.json"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/


fields2D = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        streamlines=["VX1", "VX2"],
        uids="all",
    ),
    MapMovie2D(
        "VX1",
        r"$v_x$",
        plot_coords=[0, 1],
        streamlines=["VX1", "VX2"],
        uids=[10, 15],
    ),
]
fig0 = Fig(fields2D)


def z(partvtk):
    r = partvtk.data["PART_X1"]
    theta = partvtk.data["PART_X2"]
    return r * np.cos(theta)


parts_quantities = [
    PartQuantity(
        "PART_X1",
        "PART_X1",
        uids="all",
        plot_coords=[0, 0],
    ),
    PartQuantity(
        "PART_X2",
        "PART_X2",
        uids="all",
        plot_coords=[0, 1],
    ),
    PartQuantity("z", "z", uids="all", plot_coords=[0, 2], compute=z),
]
fig1 = Fig(parts_quantities)


# Initialize context
runContext = RunContext(task, projectPath, configPath=configPath)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0, fig1])
    pipeline.run()
