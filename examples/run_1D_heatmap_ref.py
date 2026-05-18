from idefix2python import RunContext, Pipeline, Fig, SpaceTimeHeatmap
import utilities
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "1D_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/


def analytical_trajectory(t):
    Stokes0 = 1
    fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
    r0 = 2
    return utilities.integrate(fluid.vrDrift, r0, t)


quantities = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
        ref_function=analytical_trajectory,
    )
]
fig0 = Fig(
    quantities, suptitle="Dust density on heatmap, with an analytical trajectory"
)

runContext = RunContext(
    task,
    projectPath,
    frameFolder="1D_test_withref",
)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
