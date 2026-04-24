from idefix2python import RunContext, Pipeline, LineMovie1D
from pathlib import Path

projectPath = Path(__file__).parent / "data_test"
task = "1D_test"
# By default the vtks are expected to be in {projetPath}/{task}/outputs/vtks/

custom_LineMovie1Ds = [
    LineMovie1D(
        "Dust0_RHO",
        r"$\rho^\mathrm{dust}$",
        plot_coords=[0, 0],
        title="Dust0 Density",
        vmin=0,
        vmax=1.5e-4,
    )
]

LineMovie1D.suptitle = "Evolution of the density profile"
runContext = RunContext(task, projectPath)
pipeline = Pipeline(runContext, movies1D=custom_LineMovie1Ds)

pipeline.run()
