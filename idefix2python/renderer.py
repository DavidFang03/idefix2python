import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator

# from matplotlib.ticker as ticker
import numpy as np
from multiprocessing import Pool
import shutil
from pathlib import Path
from .quantities import (
    MapMovie2D,
    LineMovie1D,
    SpaceTimeHeatmap,
    OneComponentOneVariable,
    PartQuantity,
    LocalQuantity,
)
from .vtk_io import readVTK
from mpl_toolkits.axes_grid1 import make_axes_locatable


from . import tools
from .tools import LOG

matplotlib.use("Agg")

LABEL_FONTSIZE = 16
parts_cmap = plt.get_cmap("YlOrRd")

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
    """
    Every fig will have its own pattern
    runName_fig0_0000.png
    runName_fig0.mp4
    With possible args:
    [unbounded, config]
    Instead of runName, you can use a custom_name instead
    """

    def __init__(self, context, userArgs):
        self.context = context
        self.basename = context.framepath_basename

        self.userargsinfos = []
        if userArgs.noBounds:
            self.userargsinfos += ["unbounded"]

    def get_movieframe_pattern(self, figname):
        file_pattern = (
            "_".join([self.basename] + [figname] + ["*"] + self.userargsinfos) + ".png"
        )
        return str(self.context.slice1Folder / file_pattern)

    def get_movieframe_path(self, figname, frame_nb):
        filepath = self.get_movieframe_pattern(figname).replace("*", f"{frame_nb:04}")
        return filepath

    def get_timeline_path(self, figname):
        filename = "_".join([self.basename] + [figname] + self.userargsinfos) + ".png"
        return str(self.context.frameRootFolder / filename)

    def get_movie_path(self, figname):
        filename = "_".join([self.basename] + [figname] + self.userargsinfos) + ".mp4"
        return str(self.context.videosFolder / filename)


class SliceRenderer:
    def __init__(
        self,
        context,
        processor,
        figs,
        userArgs,
        options,
    ):
        self.context = context
        self.processor = processor
        self.figs = figs
        self.figsMovie = []
        self.figsTimeline = []
        self.userArgs = userArgs
        self.framesPaths = FramesPaths(context, self.userArgs)
        self.options = options
        self.doMovie = True

        if self.userArgs.doOnlyFrames or self.options.get("no_movie"):
            self.doMovie = False
        if self.userArgs.onlyMovie:
            self.doMovie = True

        self.gridInfo = self.context.gridInfo
        if self.gridInfo.active:
            self.gridInfo.apply_zoom(
                options.get("zoom", None)
            )  # initialize gridInfo.*_toshow
            self.gridInfo.get_uniform_cartesian_grid()  # for streamplot

    def set_infos(self, partsInfo):
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
                    qtyInfo.xlabel = self.gridInfo.grid_name_1
                    qtyInfo.ylabel = self.gridInfo.grid_name_2

                elif isinstance(qtyInfo, LineMovie1D):
                    qtyInfo.ylabel = qtyInfo.symbol
                    qtyInfo.ylabel = self.gridInfo.axis_name_1

                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    qtyInfo.points = (
                        self.gridInfo.X1Line
                    )  # TODO to change if user wants custom xqty
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
                    qtyInfo.xlabel = r"$t$"
                    qtyInfo.ylabel = self.gridInfo.axis_name_1

                elif isinstance(qtyInfo, PartQuantity):
                    qtyInfo.xlabel = r"$t$"
                    qtyInfo.ylabel = qtyInfo.symbol

                if qtyInfo.ref_function is not None:
                    vtktimes = (
                        self.processor.vtktimes
                    )  # TODO recover only tstart and tend from context instead
                    t_smooth = np.linspace(np.min(vtktimes), np.max(vtktimes), 10000)
                    try:
                        predicted_values = qtyInfo.ref_function(t_smooth)
                        qtyInfo.set_ref_data(t_smooth, predicted_values)
                    except Exception as e:
                        LOG(
                            f"Warning: Failed to compute ref_function for {qtyInfo.key}. Error: {e}"
                        )

                if qtyInfo.is_timeline:
                    vtktimes = self.processor.vtktimes
                    qtyInfo.points = vtktimes

                # if isinstance(qtyInfo, MapMovie2D) or isinstance(
                #     qtyInfo, SpaceTimeHeatmap
                # ):
                #     if qtyInfo.uids is not None and "alpha" not in qtyInfo.style_kwargs:
                #         qtyInfo.style_kwargs["alpha"] = 0.20
                #     elif "alpha" not in qtyInfo.style_kwargs:
                #         qtyInfo.style_kwargs["alpha"] = 1

            fig.init()

    def render(self):
        self._pre_render()
        # First render Timelines
        self.render_Frame()

        # Then render Movies frame by frame
        slice1_list = self.context.get_slice1_vtkFiles()
        vtkList = self.context.get_global_vtkFiles()
        partList = self.context.get_particles_vtkFiles()
        # If no slice1 files exist (e.g. native 2D run), fallback to global vtkList
        vtkList = slice1_list if len(slice1_list) > 0 else vtkList
        if len(self.figsMovie) > 0:
            if self.userArgs.doOnlyFrames:
                framenb_list = []
                for frame_nb in self.userArgs.doOnlyFrames:
                    if frame_nb < 0:
                        framenb_list.append(len(vtkList) + frame_nb)
                    else:
                        framenb_list.append(frame_nb)

            elif self.userArgs.every:
                framenb_list = list(range(0, len(vtkList), self.userArgs.every))

            else:
                framenb_list = enumerate(vtkList)

            render_args = zip(
                framenb_list,
                [vtkList[i] for i in framenb_list],
                [partList[i] for i in framenb_list],
            )
            with Pool(self.userArgs.jobs) as pool:
                pool.starmap(self.render_Frame, render_args)

            for figMovie in self.figsMovie:
                self.render_movie(figMovie)

    def render_movie(self, figMovie):
        if self.doMovie:
            tools.movie(
                pattern_png=self.framesPaths.get_movieframe_pattern(
                    figMovie.name
                ),  # TODO should take only the generated frames. Later PR.
                movie_path=self.framesPaths.get_movie_path(figMovie.name),
            )

    def render_Frame(self, frame_nb=None, vtkPath=None, partPath=None):
        commonvtk = None
        if vtkPath is not None:  # that means it's a movie
            figures_to_render = self.figsMovie
            VTK = readVTK(vtkPath)
            commonvtk = self.processor.process(
                datavtk=readVTK(vtkPath), partvtk=readVTK(partPath)
            )
            custom_suptitle = f"{self.context.runName}\n{Path(*vtkPath.parts[-4:])}\n$t={VTK.t[0]:.1e}$"

        else:
            custom_suptitle = None
            figures_to_render = self.figsTimeline
            frame_nb = -1

        for figure in figures_to_render:
            figure.generate_figure(custom_suptitle=custom_suptitle)
            for qtyInfo in figure.quantities:
                if isinstance(qtyInfo, MapMovie2D):
                    self._render_2D(figure, qtyInfo, commonvtk, frame_nb)
                    self.draw_particles(
                        figure,
                        part_qty=self.partsInfo.parts_Z,
                        back_qty=qtyInfo,
                        commonvtk=commonvtk,
                        frame_nb=frame_nb,
                    )
                elif isinstance(qtyInfo, LineMovie1D):
                    self._render_1D(figure, qtyInfo, commonvtk, frame_nb)
                    self.draw_particles(
                        figure,
                        part_qty=self.partsInfo.parts_X1,
                        back_qty=qtyInfo,
                        commonvtk=commonvtk,
                        frame_nb=frame_nb,
                    )
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    self._render_SpaceTimeHeatmap(figure, qtyInfo, commonvtk, frame_nb)
                    self.draw_particles(
                        figure,
                        part_qty=self.partsInfo.parts_X1,
                        back_qty=qtyInfo,
                        commonvtk=commonvtk,
                    )
                elif isinstance(qtyInfo, PartQuantity) or isinstance(
                    qtyInfo, LocalQuantity
                ):
                    self._render_TimeSeries(figure, qtyInfo, commonvtk, frame_nb)
                    self.draw_particles(figure, part_qty=qtyInfo, commonvtk=commonvtk)

                elif isinstance(qtyInfo, OneComponentOneVariable):
                    self._render_1C1V(figure, qtyInfo, commonvtk, frame_nb)
                else:
                    raise ValueError("Quantity type not supported")

                if qtyInfo.customize is not None:
                    qtyInfo.customize(figure.axes[*qtyInfo.plot_coords].ax, commonvtk)

            if vtkPath is not None:  # that means it's a movie
                png_path = self.framesPaths.get_movieframe_path(
                    figure.name, vtkPath.name[-8:-4]
                )
            else:
                png_path = self.framesPaths.get_timeline_path(figure.name)

            figure.save_and_close(png_path)

    def _draw_streamlines(self, figure, qtyInfo, data):
        method = "linear"

        # To use streamplot we need
        # - A uniformly spaced cartesian grid (xcoords, ycoords)
        # - Vector components (ux, uy) evaluated on that same grid (by linear interpolation)

        mask1 = self.gridInfo.mask1
        mask2 = self.gridInfo.mask2
        u_x1 = data[qtyInfo.streamlines[0]][mask2][:, mask1]
        u_x2 = data[qtyInfo.streamlines[1]][mask2][:, mask1]

        match self.context.geometry:
            case "cartesian":
                ux, uy = u_x1, u_x2
            case "cylindric":
                ux, uy = u_x1, u_x2
            case "polar":
                raise NotImplementedError("POLAR geometry not implemented yet")
            case "spherical":
                Theta = self.gridInfo.X2_toshow
                ux = np.sin(Theta) * u_x1 + np.cos(Theta) * u_x2
                uy = np.cos(Theta) * u_x1 - np.sin(Theta) * u_x2

        X1Line, X2Line = self.gridInfo.X1Line_toshow, self.gridInfo.X2Line_toshow
        Ux_interp = RegularGridInterpolator(
            (X1Line, X2Line),
            ux.T,
            fill_value=np.nan,
            method=method,
            bounds_error=False,
        )
        Uy_interp = RegularGridInterpolator(
            (X1Line, X2Line),
            uy.T,
            fill_value=np.nan,
            method=method,
            bounds_error=False,
        )
        pts = np.stack((self.gridInfo.X1_fromuni, self.gridInfo.X2_fromuni), axis=-1)

        figure.axes[*qtyInfo.plot_coords].ax.streamplot(
            self.gridInfo.x_uniLine,
            self.gridInfo.y_uniLine,
            Ux_interp(pts),
            Uy_interp(pts),
            **qtyInfo.streamline_kwargs,
        )

    def _draw_contours(self, figure, qtyInfo, data_mesh, cbar):
        if getattr(qtyInfo, "contours", None) is None:
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

    def _render_1D(self, figure, qty1DInfo, commonvtk, frame_nb):

        ax = figure.axes[*qty1DInfo.plot_coords].ax

        ax.plot(self.gridInfo.X1Line, commonvtk.data[qty1DInfo.key])

        # To remove?
        if len(qty1DInfo.pointsRef) > 0:
            ax.plot(
                qty1DInfo.pointsRef,
                qty1DInfo.valuesRef,
                ls="--",
                label="Analytical",
            )
            ax.legend()
        ax.set_ylim(
            *qty1DInfo.bounds
        )  # TODO bounds will be more properly handled in later PR

    def _render_2D(self, figure, qtyInfo, commonvtk, frame_nb):
        self._draw_pcolormesh(figure, qtyInfo, commonvtk.data)

    def do_timeline_stuff(self, figure, timeline, frame_nb=-1):
        """
        Draw a vertical line to show current time
        """
        ax = figure.axes[*timeline.plot_coords].ax
        if getattr(ax, "show_time_indicator", True):
            if frame_nb > 0:
                ax.axvline(x=self.processor.vtktimes[frame_nb], **timeindicator_kwargs)
                ax.show_time_indicator = False

    def _render_SpaceTimeHeatmap(self, figure, sptime, commonvtk, frame_nb=-1):
        ax = figure.axes[*sptime.plot_coords].ax

        self._draw_pcolormesh(figure, sptime)

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

        self.do_timeline_stuff(figure, sptime, frame_nb)

    def _render_TimeSeries(self, figure, timeseries, commonvtk, frame_nb=-1):
        ax = figure.axes[*timeseries.plot_coords].ax
        if isinstance(timeseries, PartQuantity) and timeseries.is_global:
            return

        if not isinstance(timeseries, PartQuantity) and not isinstance(
            timeseries, LocalQuantity
        ):
            raise NotImplementedError("only part here")
            return  # TODO some room for timevol.dat here

        ax.grid(alpha=GRID_OPACITY)
        self.do_timeline_stuff(figure, timeseries, frame_nb)

    def _render_1C1V(self, figure, onec_onev, commonvtk, frame_nb=-1):
        ax = figure.axes[*onec_onev.plot_coords].ax

        ax.plot(onec_onev.points, onec_onev.values)
        if onec_onev.xqty is None:  # that means it's a timeline
            self.do_timeline_stuff(figure, onec_onev, frame_nb)

    def draw_particles(
        self, figure, part_qty, back_qty=None, commonvtk=None, frame_nb=None
    ):
        """
        back_qty is the background. back_qty.uids are considered if back_qty is not None. Otherwise, part_qty.uids
        """
        if back_qty is None:
            ax = figure.axes[*part_qty.plot_coords].ax
            uids = part_qty.uids
        else:
            ax = figure.axes[*back_qty.plot_coords].ax
            uids = back_qty.uids

        if uids is None or len(uids) == 0:
            return

        parts_colors = None
        if back_qty is not None and back_qty.parts_color is not None:
            parts_colors = back_qty.parts_color(commonvtk)
        elif part_qty is not None and part_qty.parts_color is not None:
            parts_colors = part_qty.parts_color(commonvtk)

        if self.options.get("scatter_particles", False) and isinstance(
            back_qty, MapMovie2D
        ):
            points = part_qty.points[frame_nb, uids]
            values = part_qty.values[frame_nb, uids]
            alpha = 1
            lw = 0.5
            ax.scatter(
                points,
                values,
                c=parts_colors,
                marker="x",
                s=1,
                linewidths=0.3,
            )

        else:
            for ii, uid in enumerate(uids):
                lw = 1
                alpha = 1
                if hasattr(part_qty, "labels") and ii < len(part_qty.labels):
                    label = part_qty.labels[ii]
                else:
                    label = uid

                if parts_colors is not None:
                    color = parts_colors[ii]

                elif back_qty is not None and "color" in back_qty.parts_kwargs:
                    color = back_qty.parts_kwargs["color"]
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
                    points = np.asarray(part_qty.values)[: frame_nb + 1, uid]
                    values = np.asarray(back_qty.localqty.values)[: frame_nb + 1, uid]
                    ax.scatter(points[-1], values[-1], color=color, marker="x")
                elif back_qty is None or isinstance(back_qty, SpaceTimeHeatmap):
                    points = self.processor.vtktimes  # pre_render doesn't initialize global partquantities so part_qty.points would be empty here
                    values = np.asarray(part_qty.values)[:, uid]
                    alpha = 1
                    lw = 1
                else:
                    raise NotImplementedError(f"{back_qty} doesn't support particles")

                ax.plot(
                    points,
                    values,
                    label=label,  # TODO Show this label. Later PR.
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
            grid1 = self.gridInfo.grid1_toshow
            grid2 = self.gridInfo.grid2_toshow
            data_mesh = data[qtyInfo.key][self.gridInfo.mask2][:, self.gridInfo.mask1]

        elif isinstance(qtyInfo, SpaceTimeHeatmap):
            grid1, grid2 = np.meshgrid(
                np.asarray(self.processor.vtktimes),
                np.asarray(self.gridInfo.X1Line),
            )
            data_mesh = np.transpose(qtyInfo.values)[self.gridInfo.mask1]
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

        ax = figure.axes[*qtyInfo.plot_coords].ax

        cmesh = ax.pcolormesh(
            grid1,
            grid2,
            data_mesh,
            norm=norm,
            **qtyInfo.style_kwargs,
            # shading="gouraud",
            edgecolors="none",
            antialiased=True,
        )

        cbar = colorbar(cmesh, cbformat)
        cbar.ax.set_title(qtyInfo.symbol)

        if isinstance(qtyInfo, MapMovie2D):
            if getattr(qtyInfo, "streamlines", None):
                self._draw_streamlines(figure, qtyInfo, data)
            self._draw_contours(
                figure, qtyInfo, data_mesh, cbar
            )  # support for Spacetimeheatmap? later PR.

        return cbar


def colorbar(mappable, cbformat):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    loc = "bottom"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size="2%", pad=0.5)
    cbar = fig.colorbar(mappable, cax=cax, location=loc, format=cbformat)
    plt.sca(last_axes)
    return cbar
