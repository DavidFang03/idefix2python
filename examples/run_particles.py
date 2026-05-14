from idefix2python import RunContext, Pipeline, Fig, PartQuantity
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


px1 = PartQuantity(
    "PART_X1",
    r"$r$",
    plot_coords=[0, 0],
    ref_function=analytical_drift,
)

fig0 = Fig([px1], suptitle="A particle radial evolution, with an analytical trajectory")

runContext = RunContext(
    task,
    projectPath,
    active_directions=[0],  # necessary when there are only part*.vtk
)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
