import matplotlib.pyplot as plt
from .quantities import MapMovie2D, LineMovie1D, PartQuantity, SpaceTimeHeatmap
from .tools import LOG
import numpy as np

from . import tools

# No data should appear in Fig, Ax: they are sent by Renderer.


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

        for qtyInfo in self.quantities:
            self.axes[*qtyInfo.plot_coords].add_quantity(qtyInfo)

    def generate_figure(self, custom_suptitle=None):
        fig_width = max(8, 5 * self.columns)  # minimum width of 8
        fig_height = max(10, 5 * self.rows)  # minimum height of 10
        fig, axs = plt.subplots(
            self.rows,
            self.columns,
            figsize=(fig_width, fig_height),
            squeeze=False,
            tight_layout=True,
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
        # if self.zoom:
        #     fig.patch.set_linewidth(10)
        #     fig.patch.set_edgecolor("cornflowerblue")
        # fig.subplots_adjust(
        #     # left=0.1,
        #     # right=1 - 0.15,
        #     bottom=0.1,
        #     top=0.8,
        #     wspace=0.3,
        #     hspace=0.3,
        # )

        # TODO move to renderer?
        # if len(self.context.format_inputs_text) > 0:
        #     tools.annotateInputs(
        #         fig, self.context.format_inputs_text, padding_top=padding_top
        #     )
        self.used_coords = [list(qtyInfo.plot_coords) for qtyInfo in self.quantities]

        for i in range(self.rows):
            for j in range(self.columns):
                self.axes[i, j].generate_ax(self.fig, axs[i, j])

    def save_and_close(self, path):
        # self._clean_unused_axes()
        DPI = 350
        self.fig.savefig(path, dpi=DPI)
        plt.close(self.fig)
        LOG(f"[OK] {path}")


class Ax:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        title=None,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.vmin = None
        self.vmax = None
        self.norm = "linear"  # for heatmap only
        self.xscale = "linear"
        self.yscale = "linear"
        self.title = title
        self.qtytitles_list = []  # discarded if title is not None
        self.quantities = []
        self.is_pmesh = False

    def add_quantity(self, qtyInfo):
        self.quantities.append(qtyInfo)

        # looking for the smallest domain
        xmin, xmax = qtyInfo.xmin, qtyInfo.xmax
        vmin, vmax = qtyInfo.bounds
        if xmin is not None:
            if self.xmin is None or xmin > self.xmin:
                self.xmin = xmin
        if xmax is not None:
            if self.xmax is None or xmax < self.xmax:
                self.xmax = xmax
        if vmin is not None:
            if self.vmin is None or vmin > self.vmin:
                self.vmin = vmin
        if vmax is not None:
            if self.vmax is None or vmax < self.vmax:
                self.vmax = vmax

        # title
        title = qtyInfo.title
        if getattr(qtyInfo, "streamlines", None):
            stream_name = tools.get_streamline_name(qtyInfo.streamlines[0])
            title = rf"{title} | {stream_name} $\nearrow$"
        self.qtytitles_list.append(title)

        if isinstance(qtyInfo, MapMovie2D):
            self.is_pmesh = True

    def generate_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

        if len(self.quantities) == 0:
            self.ax.remove()
            return

        if self.is_pmesh:
            self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.vmin, self.vmax)
        if self.title is not None:
            title = ", ".join(self.qtytitles_list)
        else:
            title = self.title
        self.ax.set_title(title)
