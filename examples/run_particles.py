from idefix2python import RunContext, Pipeline, PartQuantity
import utilities
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "particles_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/


def analytical_drift(t_array):
    Stokes0 = 1
    fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
    r0 = 2
    return utilities.integrate(fluid.vrDrift, r0, t_array)


custom_partQuantities = [
    PartQuantity(
        "PART_X1",
        r"$x$",
        plot_coords=[0, 0],
        ref_function=analytical_drift,
    )
]

runContext = RunContext(
    task,
    projectPath,
    active_directions=[0],  # currently necessary for lagrangian particles.)
)
pipeline = Pipeline(runContext, partQuantities=custom_partQuantities)

pipeline.run()
