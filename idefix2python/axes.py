import matplotlib.pyplot as plt
from .quantities import (
    MapMovie2D,
    LineMovie1D,
    OneComponentOneVariable,
    SpaceTimeHeatmap,
    PartQuantity,
    LocalQuantity,
)
from .tools import LOG
import numpy as np

from . import tools

# No data should appear in Fig, Ax: they are sent by Renderer.
DPI = 550


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
            elif isinstance(qtyInfo, OneComponentOneVariable) and qtyInfo.is_movie:
                self.movie = True

            if qtyInfo.plot_coords[0] > self.rows - 1:
                self.rows = qtyInfo.plot_coords[0] + 1
            if qtyInfo.plot_coords[1] > self.columns - 1:
                self.columns = qtyInfo.plot_coords[1] + 1

        self.axes = np.empty((self.rows, self.columns), dtype="object")

    def init(self):
        """
        Only after Renderer._pre_render()
        """
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

        if self.suptitle is not None and custom_suptitle is not None:
            suptitle = f"{self.suptitle}\n{custom_suptitle}"
        elif self.suptitle is not None:
            suptitle = self.suptitle
        else:
            suptitle = custom_suptitle
        if suptitle is not None:
            fig.suptitle(suptitle)

        # TODO move to renderer? Later PR
        # if len(self.context.format_inputs_text) > 0:
        # padding_top = 0.1
        #     tools.annotateInputs(
        #         fig, self.context.format_inputs_text, padding_top=padding_top
        #     )
        self.used_coords = [list(qtyInfo.plot_coords) for qtyInfo in self.quantities]

        for i in range(self.rows):
            for j in range(self.columns):
                self.axes[i, j].generate_ax(self.fig, axs[i, j])

    def save_and_close(self, path):
        for ax in self.axes.flat:
            ax.last_pimp()
        self.fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(self.fig)
        LOG(f"[OK] {path}")


class Ax:
    def __init__(
        self,
    ):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.norm = "linear"  # for heatmap only
        self.xscale = "linear"
        self.yscale = "linear"
        self.title = None
        self.qtytitles_list = []  # discarded if title is not None
        self.quantities = []
        self.is_pmesh_grid = False
        self.active = True

    def add_quantity(self, qtyInfo):
        """
        These procedures are universal for any kind of qty.
        """
        self.quantities.append(qtyInfo)

        # looking for the largest domain
        for attr in ["xmin", "ymin", "xmax", "ymax"]:
            if getattr(qtyInfo, attr) is not None:
                if getattr(self, attr) is None:
                    setattr(self, attr, getattr(qtyInfo, attr))
                elif "min" in attr:
                    setattr(
                        self,
                        attr,
                        np.nanmin([getattr(qtyInfo, attr), getattr(self, attr)]),
                    )
                elif "max" in attr:
                    setattr(
                        self,
                        attr,
                        np.nanmax([getattr(qtyInfo, attr), getattr(self, attr)]),
                    )

        for attr in ["xscale", "yscale"]:
            if getattr(qtyInfo, attr) is not None:
                setattr(self, attr, getattr(qtyInfo, attr))

        # title
        title = qtyInfo.title
        if title is None:
            title = qtyInfo.symbol
        if getattr(qtyInfo, "streamlines", None):
            stream_name = tools.get_streamline_name(qtyInfo.streamlines[0])
            title = f"{title} \\textbar{{}} {stream_name} $\\nearrow$"
        if title is not None:
            self.qtytitles_list.append(title)

        if isinstance(qtyInfo, MapMovie2D):
            self.is_pmesh_grid = True

    def generate_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

        if len(self.quantities) == 0:
            self.active = False
            self.ax.remove()
            return

        if self.is_pmesh_grid:
            self.ax.set_aspect("equal", adjustable="box")

    def last_pimp(self):
        if not self.active:
            return

        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        title = ", ".join(self.qtytitles_list)

        self.ax.set_title(title)
