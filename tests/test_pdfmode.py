from idefix2python import RunContext, Pipeline, Fig, MapMovie2D
import numpy as np
from pathlib import Path

from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

projectPath = Path(__file__).parent / "data_test"
task = "wind2D"


runContext = RunContext(
    task,
    projectPath,
    pdf_mode=True,
)

epsilon = 0.05

Rgrid = runContext.gridInfo.X1
Thetagrid = runContext.gridInfo.X2
RLine = runContext.gridInfo.X1Line
grid1 = runContext.gridInfo.grid1
grid2 = runContext.gridInfo.grid2

Rm_max = 1e10
Am_max = 1e10
inset_pos = [0.52, 0.77, 0.2, 0.2]
x1, x2, y1, y2 = 0, 0 + 2, -2, 2  # subregion of the original image


def fourH(ax, v):
    ax.plot(RLine, 4 * RLine * epsilon, ls="--", color="darkgreen", lw=1)
    ax.plot(RLine, -4 * RLine * epsilon, ls="--", color="darkgreen", lw=1)


def post_rm(ax, v):
    grey(ax, v)
    fourH(ax, v)
    inset(ax, v)


def post_Am(ax, v):
    grey_Al(ax, v)
    fourH(ax, v)
    # inset_Am(ax, v)


def grey(ax, v):
    bg = np.where(
        v.data["Rm"] > 1e10, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
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


def grey_Al(ax, v):
    bg = np.where(
        v.data["Am"] > 1e10, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
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


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    loc = "right"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="5%", pad=0.0)
    cbar = fig.colorbar(mappable, cax=cax, location=loc)
    plt.sca(last_axes)
    return cbar


def inset(ax, v):
    bg = np.where(
        v.data["Rm"] > 1e10, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
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
    m = axins.pcolormesh(
        grid1,
        grid2,
        v.data["Rm"],
        rasterized=True,
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=1e3),
    )
    colorbar(m)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.set_aspect("equal")


def inset_Am(ax, v):
    bg = np.where(
        v.data["Am"] > 1e10, np.ones(grid1.shape) + 1, np.full(grid1.shape, np.nan)
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
        v.data["Am"],
        rasterized=True,
        cmap="inferno",
        vmin=1,
        vmax=Rm_max,
    )
    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.set_aspect("equal")


def Rm(v):
    return Rgrid ** (-1.5) * 0.05**2 / v.data["eta"]


def T(v):
    z = Rgrid * np.cos(Thetagrid)
    R = Rgrid * np.sin(Thetagrid)
    R_0 = 1
    R_floor = np.where(R > R_0, R, R_0)
    trSmoothing = 0.2
    Tdisk = epsilon**2 * R_0 / R_floor
    Tcorona = 16 * Tdisk

    temp = 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * np.tanh(
        (np.abs(z) / (epsilon * R) - 4) / (trSmoothing)
    )

    return temp
    # return v.data["PRS"] / v.data["RHO"]


quantities = [
    MapMovie2D(
        "RHO",
        r"$\rho$",
        plot_coords=[0, 0],
        title="Density",
        norm="log",
        bounds=[1e-8, 1e1],
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
        "Am",
        r"$\Lambda_\mathrm{A}$",
        plot_coords=[0, 2],
        title="Ambipolar Elsasser number",
        bounds=[1, 1e10],
        norm="log",
        customize=post_Am,
    ),
    MapMovie2D(
        "T",
        r"$T_\mathrm{target}$",
        plot_coords=[0, 3],
        title="Temperature (target)",
        compute=T,
        norm="log",
        customize=fourH,
        bounds=[1e-6, 1e-1],
    ),
]
for qt in quantities:
    qt.style_kwargs = {"cmap": "inferno"}

fig0 = Fig(quantities)


if __name__ == "__main__":
    pipeline = Pipeline(runContext, [fig0], no_movie=True)
    pipeline.run()
