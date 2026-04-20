from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "1D_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/

custom_spaceTimeHeatmaps = [
    SpaceTimeHeatmap(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
    )
]

runContext = RunContext(task, projectPath)

pipeline = Pipeline(runContext, spaceTimeHeatmaps=custom_spaceTimeHeatmaps)

pipeline.run()
