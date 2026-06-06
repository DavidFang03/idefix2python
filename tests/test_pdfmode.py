from idefix2python import RunContext, Pipeline, Fig, MapMovie2D
import numpy as np
from pathlib import Path

# projectPath = Path(__file__).parent / "data_test"
projectPath = "/home/dfang/Code/"
task = "idefix_outputs"


runContext = RunContext(
    task,
    projectPath,
    dataFolder="/home/dfang/Code/idefix_outputs/vtks",
    # pdf_mode=True,
)

epsilon = 0.05

Rgrid = runContext.gridInfo.X1
RLine = runContext.gridInfo.X1Line
grid1 = runContext.gridInfo.grid1
grid2 = runContext.gridInfo.grid2

Rm_max = 3e2
inset_pos = [0.75, 0.78, 0.2, 0.2]
x1, x2, y1, y2 = 0, 0 + 2, -2, 2  # subregion of the original image


def fourH(ax, v):
    ax.plot(RLine, 4 * RLine * epsilon, ls="--", color="white", lw=1)
    ax.plot(RLine, -4 * RLine * epsilon, ls="--", color="white", lw=1)


def post_rm(ax, v):
    grey(ax, v)
    fourH(ax, v)
    inset(ax, v)


def grey(ax, v):
    bg = np.where(
        v.data["Rm"] > 1e7, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
    )
    ax.pcolormesh(
        grid1,
        grid2,
        bg,
        rasterized=True,
        cmap="Greys",
        vmin=0,
        vmax=10,
    )


def inset(ax, v):
    bg = np.where(
        v.data["Rm"] > 1e7, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
    )
    axins = ax.inset_axes(inset_pos, xlim=(x1, x2), ylim=(y1, y2))
    axins.pcolormesh(
        grid1,
        grid2,
        bg,
        rasterized=True,
        cmap="Greys",
        vmin=0,
        vmax=10,
    )
    axins.pcolormesh(
        grid1,
        grid2,
        v.data["Rm"],
        rasterized=True,
        cmap="inferno",
        vmin=1,
        vmax=Rm_max,
    )
    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.set_aspect("equal")


def ElA(v):
    B2 = v.data["BX1"] ** 2 + v.data["BX2"] ** 2 + v.data["BX3"] ** 2
    return v.data["Am"]


def Rm(v):
    return Rgrid ** (-1.5) * 0.05**2 / v.data["eta"]


def T(v):
    return v.data["PRS"] / v.data["RHO"]


quantities = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        norm="log",
        bounds=[1e-7, 1],
        customize=fourH,
    ),
    # Create a computed field for Mach_p and draw a contour at Mach_p = 1
    MapMovie2D(
        "Rm",
        r"$\mathrm{Rm}$",
        plot_coords=[0, 1],
        title="Ohmic Reynolds number",
        compute=Rm,
        norm="log",
        bounds=[1, Rm_max],
        customize=post_rm,
    ),
    MapMovie2D(
        "elA",
        r"$\Lambda_\mathrm{A}$",
        plot_coords=[0, 2],
        title="Ambipolar Elsasser number",
        compute=ElA,
        bounds=[1, 1e10],
        norm="log",
        customize=fourH,
    ),
    MapMovie2D(
        "T",
        r"$T\mathrm{target}$",
        plot_coords=[0, 3],
        title="Temperature (target)",
        compute=T,
        norm="log",
        customize=fourH,
    ),
]
for qt in quantities:
    qt.style_kwargs = {"cmap": "inferno"}

fig0 = Fig(quantities)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0], no_movie=True)
    pipeline.run()
