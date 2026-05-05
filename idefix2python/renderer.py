import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shutil
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from .quantities import MapMovie2D, LineMovie1D, PartQuantity

from . import tools
from .tools import LOG

matplotlib.use("Agg")

LABEL_FONTSIZE = 16

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

    def _setup_figure(self, quantities_list, custom_suptitle=None):
        rows = max([qtyInfo.plot_coords[0] for qtyInfo in quantities_list]) + 1
        columns = max([qtyInfo.plot_coords[1] for qtyInfo in quantities_list]) + 1
        fig_width = max(8, 5 * columns)  # minimum width of 8
        fig_height = max(10, 5 * rows)  # minimum height of 10
        fig, axs = plt.subplots(
            rows, columns, figsize=(fig_width, fig_height), squeeze=False
        )
        padding_top = 0.1
        if custom_suptitle is None:
            for qtyInfo in quantities_list:
                if hasattr(qtyInfo, "suptitle"):
                    fig.suptitle(rf"\bfseries {qtyInfo.suptitle}", weight="bold")
                    padding_top = 0.0
                    continue
        else:
            fig.suptitle(custom_suptitle)
        if self.userArgs.zoom:
            fig.patch.set_linewidth(10)
            fig.patch.set_edgecolor("cornflowerblue")
        fig.subplots_adjust(
            left=0.1,
            right=1 - 0.05,
            bottom=0.1,
            top=0.8,
            wspace=0.5,
            hspace=0.3,
        )
        if len(self.context.format_inputs_text) > 0:
            tools.annotateInputs(
                fig, self.context.format_inputs_text, padding_top=padding_top
            )

        return fig, axs

    def _plot_pcolormesh(self, fig, ax, grid1, grid2, data, qtyInfo):

        vmin, vmax = qtyInfo.bounds
        if vmin is None or self.userArgs.noBounds:
            vmin = np.nanmin(data)
        if vmax is None or self.userArgs.noBounds:
            vmax = np.nanmax(data)

        cbar_format = FuncFormatter(tools.fmt)
        norm = Normalize(vmin=vmin, vmax=vmax)

        if qtyInfo.norm == "log":
            vmin = vmin if vmin > 0 else 1e-9
            vmax = vmax if vmax > 0 else 1e-8
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cbar_format = None
        elif qtyInfo.norm == "TwoSlopeNorm" and not self.userArgs.noBounds:
            vmin = vmin if vmin < 0 else -1e-7
            vmax = vmax if vmax > 0 else 1e-7
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        cmesh = ax.pcolormesh(
            grid1, grid2, data, cmap=qtyInfo.cmap, norm=norm, alpha=0.5
        )

        cbar = fig.colorbar(cmesh, ax=ax, format=cbar_format)
        cbar.ax.set_title(qtyInfo.symbol)

        if self.userArgs.zoom and isinstance(qtyInfo, MapMovie2D):
            ax.contourf(
                grid1,
                grid2,
                np.logical_not(self.processor.mask),
                levels=[0.5, 1],
                hatches=["////"],
                colors="none",
            )

        return cbar

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

    def _save_and_close(self, fig, path):
        fig.savefig(path, dpi=300)
        plt.close(fig)
        LOG(f"[OK] {path}")

    def _clean_unused_axes(self, axs, fields):
        rows, columns = axs.shape
        used_coords = [list(f.plot_coords) for f in fields]
        for i in range(rows):
            for j in range(columns):
                if [i, j] not in used_coords:
                    axs[i, j].remove()

    def render_2D(self, V, frame_nb, vtkPath):

        time = V.t[0]
        fig, axs = self._setup_figure(
            self.movies2D,
            custom_suptitle=f"{self.context.runName}\n{Path(*vtkPath.parts[-4:])}\n$t={time:.1e}$",
        )

        for qtyInfo in self.movies2D:
            qty = qtyInfo.key
            ax = axs[*qtyInfo.plot_coords]
            data = V.data[qty]

            grid1 = self.processor.grid1
            grid2 = self.processor.grid2

            cbar = self._plot_pcolormesh(fig, ax, grid1, grid2, data, qtyInfo)

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
                self._plot_streamlines(ax, V, qtyInfo)
            if getattr(qtyInfo, "contours", None) is not None:
                self._plot_contours(ax, data, qtyInfo, cbar)
            if qtyInfo.uids is not None:
                self._plot_particles_on_ax(
                    ax, self.processor.parts_Y, qty=qtyInfo, frame_nb=frame_nb
                )

            if color is not None:
                ax.set_title(title, color=color)
            else:
                ax.set_title(title)

        self._clean_unused_axes(axs, self.movies2D)

        slice1_name = vtkPath.name
        slice1_png_path = str(self.framesPaths.slice1_png_pattern).replace(
            "*", f"{slice1_name[:-4]}"
        )
        self._save_and_close(fig, slice1_png_path)

    def render_1D(self, V, frame_nb, vtkPath):
        if not self.movies1D:
            return

        t = V.t[0]
        fig, axs = self._setup_figure(self.movies1D, custom_suptitle=rf"$t={t:.1e}$")

        points = self.processor.X1Line

        for field1D in self.movies1D:
            key = field1D.key
            ax = axs[*field1D.plot_coords]

            ax.plot(points, V.data[key])

            if hasattr(field1D, "parts_X"):
                self._plot_particles_on_ax(
                    ax,
                    field1D.parts_X,
                    qty=field1D,
                    uids=field1D.uids,
                    frame_nb=frame_nb,
                )

            # To revove?
            if len(field1D.pointsRef) > 0:
                ax.plot(
                    field1D.points,
                    field1D.values,
                    ls="--",
                    label="Analytical",
                )
                ax.legend()
            # ax.set_xlim(self.processor.xmin, self.processor.xmax)
            ax.set_xlabel(self.processor.axis_name_1)
            ax.set_ylim(*field1D.bounds)
            ax.set_ylabel(field1D.symbol)
            ax.set_title(field1D.title)
            ax.grid()

        self._clean_unused_axes(axs, self.movies1D)

        slice1_name = vtkPath.name

        slice1_png_path = Path(
            str(self.framesPaths.slice1_png_pattern).replace(
                "*", f"1D_{slice1_name[:-4]}"
            )
        )
        self._save_and_close(fig, slice1_png_path)

    def render_SpaceTimeHeatmap(self):
        if not self.spaceTimeHeatmaps:
            return
        fig, axs = self._setup_figure(self.spaceTimeHeatmaps)

        for sptime in self.spaceTimeHeatmaps:
            ax = axs[*sptime.plot_coords]
            T, Points = np.meshgrid(
                np.asarray(self.context.outputTypes_info["vtk"].times),
                np.asarray(sptime.points),
            )

            cbar = self._plot_pcolormesh(
                fig, ax, T, Points, np.transpose(sptime.values), sptime
            )

            if hasattr(sptime, "parts_X"):
                self._plot_particles_on_ax(ax, sptime.parts_X, uids=sptime.uids)
                has_legend_items = False

            has_legend_items = False
            if len(sptime.pointsRef) > 0:
                print(sptime, sptime.pointsRef)
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
            ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
            ax.set_ylabel(self.processor.axis_name_1)
            ax.set_title(sptime.title)
            ax.grid()

        self._clean_unused_axes(axs, self.spaceTimeHeatmaps)
        self._save_and_close(fig, self.framesPaths.spacetimeheatmap_frame_path)

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
                lw = 1
            elif isinstance(qty, LineMovie1D):
                points = part_qty.values[: frame_nb + 1, uid]
                values = 0 * points
                ax.scatter(points[-1], 0, color=color, marker="x")
            elif isinstance(qty, PartQuantity):
                points = part_qty.points
                values = part_qty.values[:, uid]
                alpha = 1
                lw = 2
            else:
                raise NotImplementedError(f"{qty} doesn't support particles")
            ax.plot(points, values, label=label, color=color, lw=lw, alpha=alpha)
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

    def render_timeSeries(self):
        if not self.partQuantities:
            return
        notglobal_partquantities = [v for v in self.partQuantities if not v.is_global]
        if len(notglobal_partquantities) > 0:
            fig, axs = self._setup_figure(notglobal_partquantities)
            for qtyInfo in notglobal_partquantities:
                ax = axs[*qtyInfo.plot_coords]
                self._plot_particles_on_ax(ax, qtyInfo)

                ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
                ax.set_ylabel(qtyInfo.symbol)
                ax.set_title(qtyInfo.title)
                ax.grid()
            self._clean_unused_axes(axs, notglobal_partquantities)
            self._save_and_close(fig, self.framesPaths.timeSeries_frame_path)
