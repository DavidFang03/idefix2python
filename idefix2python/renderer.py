import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

# from matplotlib.ticker as ticker
import numpy as np
from multiprocessing import Pool
import shutil
from pathlib import Path
from .quantities import MapMovie2D, LineMovie1D, SpaceTimeHeatmap, PartQuantity
from .vtk_io import readVTK
from mpl_toolkits.axes_grid1 import make_axes_locatable


from . import tools
from .tools import LOG

matplotlib.use("Agg")

LABEL_FONTSIZE = 16
DPI = 350
parts_cmap = plt.get_cmap("Pastel1")

timeindicator_kwargs = {"lw": 1, "ls": "--", "alpha": 0.8}
GRID_OPACITY = 0.1

plt.style.use("dark_background")


# Check if latex is in the system PATH
if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True})
else:
    LOG("Warning: LaTeX not found. Using standard Matplotlib fonts.")
    plt.rcParams.update({"text.usetex": False})

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    }
)
plt.rcParams["hatch.color"] = "gray"
plt.rcParams["hatch.linewidth"] = 0.5
plt.rcParams["font.size"] = 12


class FramesPaths:
    def __init__(self, context, userArgs):

        filenameinfos = []
        if userArgs.zoom:
            filenameinfos += [f"zoom{userArgs.zoom}"]

        if userArgs.noBounds:
            filenameinfos += ["unbounded"]
        elif context.configPath is not None:
            filenameinfos += ["config"]

        slice1_png_pattern = "_".join(["*"] + filenameinfos) + ".png"
        slice1Movie_path = "_".join([context.runName] + filenameinfos) + ".mp4"

        self.slice1_png_pattern = context.slice1Folder / slice1_png_pattern
        self.slice1_video_path = context.videosFolder / slice1Movie_path

        self.timeline_frame_path = context.frameRootFolder / f"{context.runName}.png"


class SliceRenderer:
    def __init__(
        self,
        context,
        processor,
        figs,
        userArgs,
    ):
        self.context = context
        self.processor = processor
        self.figs = figs
        self.figsMovie = []
        self.figsTimeline = []
        self.userArgs = userArgs
        self.framesPaths = FramesPaths(context, self.userArgs)

        if self.userArgs.doOnlyFrames and not self.userArgs.onlyMovie:
            self.doMovie = False

    def set_infos(self, gridInfo, partsInfo):
        self.gridInfo = gridInfo
        self.partsInfo = partsInfo

    def _pre_render(self):
        """
        Before rendering, making sure every quantity has the seatbelt fastened
        """
        for fig in self.figs:
            if fig.movie:
                self.figsMovie.append(fig)
            else:
                self.figsTimeline.append(fig)

            for qtyInfo in fig.quantities:
                if isinstance(qtyInfo, MapMovie2D):
                    for attr in ["xmin", "xmax", "ymin", "ymax"]:
                        value = (
                            getattr(qtyInfo, attr)
                            if getattr(qtyInfo, attr) is not None
                            else getattr(self.gridInfo, attr)
                        )
                        setattr(qtyInfo, attr, value)
                    self.xlabel, self.ylabel = (
                        self.gridInfo.grid_name_1,
                        self.gridInfo.grid_name_2,
                    )
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    qtyInfo.xmin = (
                        qtyInfo.xmin
                        if qtyInfo.xmin is not None
                        else np.min(self.processor.vtktimes)
                    )
                    qtyInfo.xmax = (
                        qtyInfo.xmax
                        if qtyInfo.xmax is not None
                        else np.max(self.processor.vtktimes)
                    )
                    qtyInfo.ymin = (
                        qtyInfo.ymin
                        if qtyInfo.ymin is not None
                        else np.nanmin(qtyInfo.points)
                    )
                    qtyInfo.ymax = (
                        qtyInfo.ymax
                        if qtyInfo.ymax is not None
                        else np.nanmax(qtyInfo.points)
                    )

                    qtyInfo.xlabel, qtyInfo.ylabel = r"$t$", self.gridInfo.grid_name_1
                elif isinstance(qtyInfo, PartQuantity):
                    qtyInfo.xlabel, qtyInfo.ylabel = (
                        r"$t$",
                        qtyInfo.symbol,
                    )  # TODO problems?

                fig.init()

    def render(self):
        self._pre_render()
        # First render Timelines
        self.render_Frame()

        # Then render Movies frame by frame
        slice1_list = self.context.get_slice1_vtkFiles()
        self.vtkList = self.context.get_global_vtkFiles()
        if len(self.figsMovie) > 0:
            # If no slice1 files exist (e.g. native 2D run), fallback to global vtkList
            render_list = slice1_list if len(slice1_list) > 0 else self.vtkList

            if self.userArgs.doOnlyFrames:
                render_list = [render_list[i] for i in self.userArgs.doOnlyFrames]
            with Pool(self.userArgs.jobs) as pool:
                pool.starmap(self.render_Frame, enumerate(render_list))

            if len(self.figsMovie) > 0:
                self.render_movie()

    def render_movie(self):
        tools.movie(
            pattern_png=self.framesPaths.slice1_png_pattern,  # TODO should be one pattern for every fig
            movie_path=self.framesPaths.slice1_video_path,
        )

    def render_Frame(self, frame_nb=None, vtkPath=None):

        if vtkPath is not None:  # that means it's a movie
            figures_to_render = self.figsMovie
            VTK = readVTK(vtkPath)
            self.processor.process(VTK)
            custom_suptitle = f"{self.context.runName}\n{Path(*vtkPath.parts[-4:])}\n$t={VTK.t[0]:.1e}$"

        else:
            custom_suptitle = None
            figures_to_render = self.figsTimeline
            frame_nb = -1

        for figure in figures_to_render:
            figure.generate_figure(custom_suptitle=custom_suptitle)
            for qtyInfo in figure.quantities:
                if isinstance(qtyInfo, MapMovie2D):
                    self._render_2D(figure, qtyInfo, VTK.data, frame_nb)
                elif isinstance(qtyInfo, LineMovie1D):
                    self._render_1D(figure, qtyInfo, VTK.data, frame_nb)
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    self._render_SpaceTimeHeatmap(figure, qtyInfo, frame_nb)
                elif isinstance(qtyInfo, PartQuantity):
                    self._render_TimeSeries(figure, qtyInfo, frame_nb)

            if vtkPath is not None:  # that means it's a movie
                png_path = str(self.framesPaths.slice1_png_pattern).replace(
                    "*", f"{figure.name}_{vtkPath.name[:-4]}"
                )
            else:
                png_path = (
                    self.framesPaths.timeline_frame_path.parent
                    / f"{figure.name}_{self.framesPaths.timeline_frame_path.name}"
                )
            figure.save_and_close(png_path)

    def _draw_streamlines(self, figure, qtyInfo, data):
        lw_streamline = 0.2
        density_streamline = [1, 2]
        arrowstyle_streamline = "->"

        if (
            isinstance(qtyInfo.streamlines, (list, tuple))
            and len(qtyInfo.streamlines) == 2
        ):
            u_key1, u_key2 = qtyInfo.streamlines
            if u_key1 in data and u_key2 in data:
                ux, uz = tools.convertVector_toXZ(
                    data[u_key1],
                    data[u_key2],
                    self.gridInfo.X1,
                    self.gridInfo.X2,
                    self.context.geometry,
                )
                x_coords, z_coords, Ux_uni, Uy_uni = tools.get_streamplot_data(
                    self.gridInfo.X1Line,
                    self.gridInfo.X2Line,
                    ux,
                    uz,
                    self.context.geometry,
                    self.gridInfo.xmax,
                )
                figure.axes[*qtyInfo.plot_coords].ax.streamplot(
                    x_coords,
                    z_coords,
                    Ux_uni,
                    Uy_uni,
                    density=density_streamline,
                    linewidth=lw_streamline,
                    arrowstyle=arrowstyle_streamline,
                    color=qtyInfo.streamline_color,
                )

        else:
            raise Exception(
                f"Invalid streamline configuration: {qtyInfo.streamlines}. Expected a list/tuple of length 2."
            )

    def _draw_contours(self, figure, qtyInfo, data_mesh, cbar):
        if not getattr(qtyInfo, "contours", None):
            return

        levels = figure.axes[*qtyInfo.plot_coords].ax.contour(
            self.gridInfo.grid1,
            self.gridInfo.grid2,
            data_mesh,
            qtyInfo.contours,
            alpha=0.5,
            colors=[qtyInfo.contour_color],
            linewidths=[1.5],
        )
        cbar.add_lines(levels)

    def _render_1D(self, figure, qty1DInfo, data, frame_nb):

        ax = figure.axes[*qty1DInfo.plot_coords].ax

        ax.plot(self.gridInfo.X1Line, data[qty1DInfo.key])

        self.draw_particles(
            figure,
            part_qty=self.partsInfo.parts_X1,
            back_qty=qty1DInfo,
            frame_nb=frame_nb,
        )

        # To remove?
        if len(qty1DInfo.pointsRef) > 0:
            ax.plot(
                qty1DInfo.points,
                qty1DInfo.values,
                ls="--",
                label="Analytical",
            )
            ax.legend()
        # ax.set_xlim(self.processor.xmin, self.processor.xmax)
        # ax.set_xlabel(self.processor.axis_name_1)
        # ax.set_ylim(*qty1DInfo.bounds)
        # ax.set_ylabel(qty1DInfo.symbol)
        # ax.set_title(qty1DInfo.title)
        # ax.grid()

    def _render_2D(self, figure, qtyInfo, data, frame_nb):
        self._draw_pcolormesh(figure, qtyInfo, data)
        self.draw_particles(
            figure,
            part_qty=self.partsInfo.parts_Z,
            back_qty=qtyInfo,
            frame_nb=frame_nb,
        )

    def do_timeline_stuff(self, figure, timeline, frame_nb=-1):
        ax = figure.axes[*timeline.plot_coords].ax
        if frame_nb > 0:
            ax.axvline(x=self.processor.vtktimes[frame_nb], **timeindicator_kwargs)

    def _render_SpaceTimeHeatmap(self, figure, sptime, frame_nb=-1):
        ax = figure.axes[*sptime.plot_coords].ax

        self._draw_pcolormesh(figure, sptime)

        self.draw_particles(
            figure,
            part_qty=self.partsInfo.parts_X1,
            back_qty=sptime,
        )

        has_legend_items = False
        if len(sptime.pointsRef) > 0:
            plot_kwargs = {}
            if hasattr(sptime.ref_function, "plot_kwargs"):
                plot_kwargs = sptime.ref_function.plot_kwargs
                if "zorder" not in plot_kwargs:
                    plot_kwargs["zorder"] = 3
                if "label" in plot_kwargs:
                    has_legend_items = True
            ax.plot(
                sptime.pointsRef,
                sptime.valuesRef,
                **plot_kwargs,
            )

        if has_legend_items:
            ax.legend()

        ax.set_ylabel(self.gridInfo.native_grid_name_1)
        self.do_timeline_stuff(figure, sptime, frame_nb)

    def _render_TimeSeries(self, figure, timeseries, frame_nb=-1):
        ax = figure.axes[*timeseries.plot_coords].ax
        if isinstance(timeseries, PartQuantity) and timeseries.is_global:
            return

        if isinstance(timeseries, PartQuantity):
            self.draw_particles(
                figure,
                part_qty=timeseries,
            )
        else:
            raise NotImplementedError("only part here")
            return  # TODO some place for timevol.dat here

        ax.grid(alpha=GRID_OPACITY)
        ax.set_ylabel(timeseries.symbol)
        self.do_timeline_stuff(figure, timeseries, frame_nb)

    def draw_particles(self, figure, part_qty, back_qty=None, frame_nb=None):
        """
        back_qty is the background. If back_qty.uids are accounted if back_qty is not None. Otherwise, part_qty.uids
        """
        if back_qty is None:
            ax = figure.axes[*part_qty.plot_coords].ax
            uids = part_qty.uids
        else:
            ax = figure.axes[*back_qty.plot_coords].ax
            uids = back_qty.uids

        # if uids == "all" or len(uids) == 0: # ???
        if uids is None:
            return

        for ii, uid in enumerate(uids):
            lw = 1
            alpha = 1
            if hasattr(part_qty, "labels") and ii < len(part_qty.labels):
                label = part_qty.labels[ii]
            else:
                label = uid

            # TODO Make this part applicable to any kwargs, not just color/label
            if back_qty is not None and "color" in back_qty.parts_kwargs:
                color = back_qty.parts_kwargs["color"]
            elif hasattr(part_qty, "colors") and ii < len(part_qty.colors):
                color = part_qty.colors[ii]
            else:
                color = parts_cmap(ii / max(1, len(uids) - 1))

            if isinstance(back_qty, MapMovie2D):
                points = part_qty.points[: frame_nb + 1, uid]
                values = part_qty.values[: frame_nb + 1, uid]
                alpha = 1
                lw = 0.5
                ax.scatter(
                    points[-1],
                    values[-1],
                    color=color,
                    marker="x",
                    s=1,
                    linewidths=0.3,
                )
            elif isinstance(back_qty, LineMovie1D):
                points = part_qty.values[: frame_nb + 1, uid]
                values = 0 * points
                ax.scatter(points[-1], 0, color=color, marker="x")
            elif back_qty is None or isinstance(back_qty, SpaceTimeHeatmap):
                points = part_qty.points
                values = part_qty.values[:, uid]
                alpha = 1
                lw = 2
            else:
                raise NotImplementedError(f"{back_qty} doesn't support particles")

            ax.plot(
                points,
                values,
                label=label,
                color=color,
                lw=lw,
                alpha=alpha,
                marker="8",
                markersize=0.2,
            )

        if len(part_qty.pointsRef) > 0:
            ax.plot(
                part_qty.pointsRef, part_qty.valuesRef, ls="--", lw=2, label="Predicted"
            )

    def _draw_pcolormesh(self, figure, qtyInfo, data=None):
        """
        For MapMovie2D, passing the entire data is necessary for streamlines.
        """

        if isinstance(qtyInfo, MapMovie2D):
            grid1 = self.gridInfo.grid1
            grid2 = self.gridInfo.grid2
            data_mesh = data[qtyInfo.key]

        elif isinstance(qtyInfo, SpaceTimeHeatmap):
            grid1, grid2 = np.meshgrid(
                np.asarray(self.processor.vtktimes),
                np.asarray(self.gridInfo.X1Line),
            )
            data_mesh = np.transpose(qtyInfo.values)
        vmin, vmax = qtyInfo.bounds
        if vmin is None or self.userArgs.noBounds:
            vmin = np.nanmin(data_mesh)
        if vmax is None or self.userArgs.noBounds:
            vmax = np.nanmax(data_mesh)

        cbformat = matplotlib.ticker.ScalarFormatter()
        cbformat.set_scientific("%.2e")
        cbformat.set_powerlimits((-2, 12))
        cbformat.set_useMathText(True)

        norm = Normalize(vmin=vmin, vmax=vmax)

        if qtyInfo.norm == "log":
            vmin = vmin if vmin > 0 else 1e-9
            vmax = vmax if vmax > 0 else 1e-8
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cbformat = None
        elif qtyInfo.norm == "TwoSlopeNorm" and not self.userArgs.noBounds:
            vmin = vmin if vmin < 0 else -1e-7
            vmax = vmax if vmax > 0 else 1e-7
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        if qtyInfo is not None:
            alpha = 0.20
        else:
            alpha = 1

        ax = figure.axes[*qtyInfo.plot_coords].ax

        cmesh = ax.pcolormesh(
            grid1,
            grid2,
            data_mesh,
            cmap=qtyInfo.cmap,
            norm=norm,
            alpha=alpha,  # TODO more customization
            antialiased=True,  # to remove artefacts
        )

        cbar = colorbar(cmesh, cbformat)
        cbar.ax.set_title(qtyInfo.symbol)

        if isinstance(qtyInfo, MapMovie2D):
            if getattr(qtyInfo, "streamlines", None):
                self._draw_streamlines(figure, qtyInfo, data)
            self._draw_contours(
                figure, qtyInfo, data_mesh, cbar
            )  # support for Spacetimeheatmap? later PR.

        if self.userArgs.zoom and isinstance(qtyInfo, MapMovie2D):
            ax.contourf(
                grid1,
                grid2,
                np.logical_not(self.gridInfo.mask),
                levels=[0.5, 1],
                hatches=["////"],
                colors="none",
            )

        return cbar


def colorbar(mappable, cbformat):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    loc = "bottom"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="5%", pad=0.75)
    cbar = fig.colorbar(mappable, cax=cax, location=loc, format=cbformat)
    plt.sca(last_axes)
    return cbar
