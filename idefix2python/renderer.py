import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path
from .quantities import MapMovie2D, LineMovie1D, SpaceTimeHeatmap, PartQuantity
from .vtk_io import readVTK

from . import tools
from .tools import LOG

matplotlib.use("Agg")

LABEL_FONTSIZE = 16
DPI = 350

lw_streamline = 0.2
density_streamline = [1, 2]
arrowstyle_streamline = "->"

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


class SliceRenderer:
    def __init__(
        self,
        context,
        processor,
        spaceTimeHeatmaps,
        movies1D,
        movies2D,
        partQuantities,
        userArgs,
        framesPaths,
    ):
        self.context = context
        self.processor = processor
        self.spaceTimeHeatmaps = spaceTimeHeatmaps
        self.movies2D = movies2D
        self.movies1D = movies1D
        self.partQuantities = partQuantities
        self.userArgs = userArgs
        self.framesPaths = framesPaths

    def _plot_streamlines(self, ax, V, qtyInfo):
        data = V.data
        if (
            isinstance(qtyInfo.streamlines, (list, tuple))
            and len(qtyInfo.streamlines) == 2
        ):
            u_key1, u_key2 = qtyInfo.streamlines
            if u_key1 in data and u_key2 in data:
                ux, uz = tools.convertVector_toXZ(
                    data[u_key1],
                    data[u_key2],
                    self.processor.X1,
                    self.processor.X2,
                    self.context.geometry,
                )
                x_coords, z_coords, Ux_uni, Uy_uni = tools.get_streamplot_data(
                    self.processor.X1Line,
                    self.processor.X2Line,
                    ux,
                    uz,
                    self.context.geometry,
                    self.processor.xmax,
                )
                ax.streamplot(
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

    def _plot_contours(self, ax, data, qtyInfo, cbar):
        levels = ax.contour(
            self.processor.grid1,
            self.processor.grid2,
            data,
            qtyInfo.contours,
            alpha=0.5,
            colors=[qtyInfo.contour_color],
            linewidths=[1.5],
        )
        cbar.add_lines(levels)

    def render_MovieFrame(self, frame_nb, vtkPath):
        VTK = readVTK(vtkPath)
        self.processor.process(VTK)

        for figure in self.processor.figsMovie:
            time = VTK.t[0]
            figure.setup_figure(
                custom_suptitle=f"{self.context.runName}\n{Path(*vtkPath.parts[-4:])}\n$t={time:.1e}$"
            )
            for qtyInfo in figure.quantities:
                if isinstance(qtyInfo, MapMovie2D):
                    self._render_2D(figure, qtyInfo, VTK, frame_nb)
                elif isinstance(qtyInfo, LineMovie1D):
                    self._render_1D(figure, qtyInfo, VTK, frame_nb)
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    self._render_SpaceTimeHeatmap(figure, qtyInfo)
                elif isinstance(qtyInfo, PartQuantity):
                    self._render_TimeSeries(figure, qtyInfo)

            slice1_name = vtkPath.name
            slice1_png_path = str(self.framesPaths.slice1_png_pattern).replace(
                "*", f"{figure.name}_{slice1_name[:-4]}"
            )
            figure.save_and_close(slice1_png_path)

    def render_TimelineFrame(self):
        for figure in self.processor.figsTimeline:
            figure.setup_figure()
            for qtyInfo in figure.quantities:
                if isinstance(qtyInfo, SpaceTimeHeatmap):
                    self._render_SpaceTimeHeatmap(figure, qtyInfo)
                if isinstance(qtyInfo, PartQuantity):
                    self._render_TimeSeries(figure, qtyInfo)
                # TODO support movies by showing final frame?

            timeseriespath = (
                self.framesPaths.timeline_frame_path.parent
                / f"{figure.name}_{self.framesPaths.timeline_frame_path.name}"
            )
            figure.save_and_close(timeseriespath)

    def _render_2D(self, figure, qtyInfo, VTK, frame_nb):

        data = VTK.data[qtyInfo.key]

        ax = figure.axes[*qtyInfo.plot_coords].ax

        grid1 = self.processor.grid1
        grid2 = self.processor.grid2
        cbar = figure.plot_pcolormesh(
            grid1, grid2, data, qtyInfo, zoom=self.userArgs.zoom
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.processor.xmin, self.processor.xmax)
        ax.set_ylim(self.processor.ymin, self.processor.ymax)

        ax.set_xlabel(self.processor.grid_name_1)
        if qtyInfo.plot_coords[1] == 0:
            ax.set_ylabel(self.processor.grid_name_2)

        title = qtyInfo.title
        color = None

        if getattr(qtyInfo, "streamlines", None):
            color = qtyInfo.streamline_color
            stream_name = tools.get_streamline_name(qtyInfo.streamlines[0])
            title = rf"{title} | {stream_name} $\nearrow$"
            self._plot_streamlines(ax, VTK, qtyInfo)
        if getattr(qtyInfo, "contours", None) is not None:
            self._plot_contours(ax, data, qtyInfo, cbar)
        if qtyInfo.uids is not None:
            self._plot_particles_on_ax(
                ax,
                self.processor.parts_Z,
                qty=qtyInfo,
                frame_nb=frame_nb,
                uids=qtyInfo.uids,
            )

        if color is not None:
            ax.set_title(title, color=color)
        else:
            ax.set_title(title)

    def _render_1D(self, figure, qty1DInfo, VTK, frame_nb):

        ax = figure.axes[*qty1DInfo.plot_coords].ax

        points = self.processor.X1Line
        ax.plot(points, VTK.data[qty1DInfo.key])

        if qty1DInfo.uids is not None:
            self._plot_particles_on_ax(
                ax,
                self.processor.parts_X1,
                qty=qty1DInfo,
                uids=qty1DInfo.uids,
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

    def _render_SpaceTimeHeatmap(self, figure, sptime):
        ax = figure.axes[*sptime.plot_coords].ax

        T, Points = np.meshgrid(
            np.asarray(self.processor.vtktimes),
            np.asarray(sptime.points),
        )

        cbar = figure.plot_pcolormesh(
            T, Points, np.transpose(sptime.values), sptime, self.userArgs.zoom
        )

        if sptime.uids is not None:
            self._plot_particles_on_ax(ax, self.processor.parts_X1, uids=sptime.uids)
            has_legend_items = False

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

        ymin = np.min(sptime.points)
        ymax = np.max(sptime.points)
        xmin = sptime.xmin
        xmax = sptime.xmax
        if sptime.ymin is not None:
            ymin = sptime.ymin
        if sptime.ymax is not None:
            ymax = sptime.ymax
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        cbar.ax.set_title(sptime.symbol)
        # ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
        # ax.set_ylabel(self.processor.axis_name_1)
        # ax.set_title(sptime.title)
        # ax.grid()

    def _render_TimeSeries(self, figure, timeseries):
        if isinstance(timeseries, PartQuantity) and timeseries.is_global:
            return

        ax = figure.axes[*timeseries.plot_coords].ax
        if isinstance(timeseries, PartQuantity):
            self._plot_particles_on_ax(ax, timeseries)
        else:
            return  # TODO some place for timevol.dat here
            # ax.set_ylim(*qtyInfo.bounds)
            # if qtyInfo.scale == "log":
            #     ax.set_yscale("log")

            # ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
            # ax.set_ylabel(qtyInfo.symbol)
            # ax.set_title(qtyInfo.title)
            # ax.grid()

    def _plot_particles_on_ax(self, ax, part_qty, qty=None, frame_nb=None, uids=None):
        has_legend_items = False
        cmap = plt.get_cmap("Pastel1")

        if uids is None:
            uids = part_qty.uids
        uids = (
            self.context.all_particles_uids
            if (uids == "all" or len(uids) == 0)
            else uids
        )
        for ii, uid in enumerate(uids):
            lw = 1
            alpha = 1
            if hasattr(part_qty, "labels") and ii < len(part_qty.labels):
                label = part_qty.labels[ii]
            else:
                label = uid

            if hasattr(part_qty, "colors") and ii < len(part_qty.colors):
                color = part_qty.colors[ii]
            else:
                color = cmap(ii / max(1, len(uids) - 1))

            if isinstance(qty, MapMovie2D):
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
            elif isinstance(qty, LineMovie1D):
                points = part_qty.values[: frame_nb + 1, uid]
                values = 0 * points
                ax.scatter(points[-1], 0, color=color, marker="x")
            elif qty is None:
                points = part_qty.points
                values = part_qty.values[:, uid]
                alpha = 1
                lw = 2
            else:
                raise NotImplementedError(f"{qty} doesn't support particles")
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
            has_legend_items = True

        if len(part_qty.pointsRef) > 0:
            ax.plot(
                part_qty.pointsRef, part_qty.valuesRef, ls="--", lw=2, label="Predicted"
            )
            has_legend_items = True

        if has_legend_items and not part_qty.is_global:
            ax.legend()
        # if has_legend_items:
        #     loc = "best"
        #     if qty.is_for2D:
        #         loc = "lower right"
        #     ax.legend(loc=loc)
