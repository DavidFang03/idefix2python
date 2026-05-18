from idefix2python import RunContext, Pipeline, Fig, SpaceTimeHeatmap
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "1D_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/

quantities = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
    )
]

fig0 = Fig(quantities, suptitle="Dust density on a heatmap")

runContext = RunContext(task, projectPath)

if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0])
    pipeline.run()
