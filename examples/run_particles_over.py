from idefix2python import RunContext, Pipeline, Fig, PartQuantity, SpaceTimeHeatmap
import utilities
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "particles_over_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/
# In this example, the vtks/ folder contains both part*.vtk and data*.vtk


def analytical_trajectory(t):
    z0 = 0.1
    fluid = utilities.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=1, z0=z0)
    return utilities.solve_2nd_order_ode(fluid.azSettling, z0, 0, t)


analytical_trajectory.plot_kwargs = {"ls": "--", "color": "cyan", "lw": 2}

z_part = PartQuantity(
    "PART_X3",
    r"$z^\mathrm{part}$",
    plot_coords=[0, 0],
)
dust0_rho = SpaceTimeHeatmap(
    "Dust0_RHO",
    r"$\rho^\mathrm{dust}$",
    plot_coords=[0, 0],
    uids="all",
    ref_function=analytical_trajectory,
)

fig0 = Fig(
    [z_part, dust0_rho],
    suptitle="Dust density: presureless fluid, particles, and an analytical trajectory",
)

runContext = RunContext(
    task,
    projectPath,
    active_directions=[2],  # necessary when there are only part*.vtk
)
pipeline = Pipeline(runContext, [fig0])

pipeline.run()
