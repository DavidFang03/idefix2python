from idefix2python import (
    RunContext,
    Pipeline,
    Fig,
    SpaceTimeHeatmap,
    OneComponentOneVariable,
)
from pathlib import Path
import numpy as np

projectPath = Path(__file__).parent / "data_examples"
task = "1D_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/


def maxrho(v):
    return v.x[np.argmax(v.data["Dust0_RHO"])]


quantities = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
    ),
    OneComponentOneVariable(
        "max_rhodust", "max_rhodust", plot_coords=[0, 0], compute=maxrho
    ),
]

fig0 = Fig(quantities, suptitle="Dust density on a heatmap")

runContext = RunContext(task, projectPath)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
