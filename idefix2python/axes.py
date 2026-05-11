import matplotlib.pyplot as plt
from .quantities import MapMovie2D, LineMovie1D, PartQuantity, SpaceTimeHeatmap
from .tools import LOG
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from . import tools


class Fig:
    counter = -1

    def __init__(self, quantities, suptitle=None):
        Fig.counter += 1
        self.name = f"fig{Fig.counter}"
        self.quantities = quantities
        self.suptitle = suptitle

        self.axesMovie = []
        self.axesTimeline = []
        self.movie = False
        self.rows = 1
        self.columns = 1
        for qtyInfo in quantities:
            if isinstance(qtyInfo, LineMovie1D) or isinstance(qtyInfo, MapMovie2D):
                self.movie = True
                self.axesMovie.append(qtyInfo)
            elif isinstance(qtyInfo, PartQuantity) or isinstance(
                qtyInfo, SpaceTimeHeatmap
            ):
                self.axesTimeline.append(qtyInfo)
            else:
                raise ValueError("Quantity type not supported")

            if qtyInfo.plot_coords[0] > self.rows - 1:
                self.rows = qtyInfo.plot_coords[0] + 1
            if qtyInfo.plot_coords[1] > self.columns - 1:
                self.columns = qtyInfo.plot_coords[1] + 1

        self.axes = np.empty((self.rows, self.columns), dtype="object")
        for i in range(self.rows):
            for j in range(self.columns):
                self.axes[i, j] = Ax()

    def setup_figure(self, custom_suptitle=None):
        fig_width = max(8, 5 * self.columns)  # minimum width of 8
        fig_height = max(10, 5 * self.rows)  # minimum height of 10
        fig, axs = plt.subplots(
            self.rows,
            self.columns,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        self.fig = fig
        padding_top = 0.1
        if self.suptitle is None:
            if custom_suptitle is not None:
                fig.suptitle(custom_suptitle)
            for qtyInfo in self.quantities:  # TODO remove?
                if hasattr(qtyInfo, "suptitle"):
                    fig.suptitle(rf"\bfseries {qtyInfo.suptitle}", weight="bold")
                    padding_top = 0.0
                    continue
        else:
            fig.suptitle(self.suptitle)
        # if self.userArgs.zoom:
        #     fig.patch.set_linewidth(10)
        #     fig.patch.set_edgecolor("cornflowerblue")
        fig.subplots_adjust(
            left=0.1,
            right=1 - 0.05,
            bottom=0.1,
            top=0.8,
            wspace=0.5,
            hspace=0.3,
        )

        # TODO move to renderer?
        # if len(self.context.format_inputs_text) > 0:
        #     tools.annotateInputs(
        #         fig, self.context.format_inputs_text, padding_top=padding_top
        #     )

        for qtyInfo in self.quantities:
            self.axes[*qtyInfo.plot_coords].set_figax(
                self.fig, axs[*qtyInfo.plot_coords]
            )

    def _clean_unused_axes(self):
        rows, columns = self.axes.shape
        used_coords = [list(qtyInfo.plot_coords) for qtyInfo in self.quantities]
        for i in range(self.rows):
            for j in range(self.columns):
                if [i, j] not in used_coords:
                    self.axes[i, j].remove()

    def save_and_close(self, path):
        self._clean_unused_axes()
        DPI = 350
        self.fig.savefig(path, dpi=DPI)
        plt.close(self.fig)
        LOG(f"[OK] {path}")

    def plot_pcolormesh(self, grid1, grid2, data, qtyInfo, zoom):

        ax = self.axes[*qtyInfo.plot_coords].ax
        vmin, vmax = qtyInfo.bounds
        if vmin is None:
            # if vmin is None or self.userArgs.noBounds: # TODO
            vmin = np.nanmin(data)
        if vmax is None:
            # if vmax is None or self.userArgs.noBounds: # TODO
            vmax = np.nanmax(data)

        cbar_format = FuncFormatter(tools.fmt)
        norm = Normalize(vmin=vmin, vmax=vmax)

        if qtyInfo.norm == "log":
            vmin = vmin if vmin > 0 else 1e-9
            vmax = vmax if vmax > 0 else 1e-8
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cbar_format = None
        elif qtyInfo.norm == "TwoSlopeNorm":
            # elif qtyInfo.norm == "TwoSlopeNorm" and not self.userArgs.noBounds: # TODO
            vmin = vmin if vmin < 0 else -1e-7
            vmax = vmax if vmax > 0 else 1e-7
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        if qtyInfo is not None:
            alpha = 0.20
        else:
            alpha = 1

        cmesh = ax.pcolormesh(
            grid1,
            grid2,
            data,
            cmap=qtyInfo.cmap,
            norm=norm,
            alpha=alpha,  # TODO more customization
            antialiased=True,  # to remove artefacts
        )

        cbar = self.fig.colorbar(cmesh, ax=ax, format=cbar_format)
        cbar.ax.set_title(qtyInfo.symbol)

        if zoom and isinstance(qtyInfo, MapMovie2D):
            ax.contourf(
                grid1,
                grid2,
                np.logical_not(self.processor.mask),
                levels=[0.5, 1],
                hatches=["////"],
                colors="none",
            )

        return cbar


class Ax:
    def __init__(self, vmin=None, vmax=None, norm="linear", title=None):
        self.vmin = vmin
        self.vmax = vmax
        self.norm = norm
        self.title = title

    def set_figax(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def remove(self):
        self.ax.remove()


# class AxMovie(Ax):
#     def __init__(self, fig, vmin=None, vmax=None, norm="linear", title=None):
#         super().__init__(fig, vmin, vmax, norm, title)


# class AxTimeline(Ax):
#     def __init__(self, fig, vmin=None, vmax=None, norm="linear", title=None):
#         super().__init__(fig, vmin, vmax, norm, title)
