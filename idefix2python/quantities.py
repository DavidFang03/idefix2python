from itertools import count
from .tools import LOG

DEFAULT_CMAP = "berlin"
DPI = 300


class Data:
    """
    Base class for all data quantities in the pipeline.

    :param key: Unique identifier for the field.
    :type key: str
    :param symbol: Symbol for labels (e.g., r"$\rho$").
    :type symbol: str
    :param plot_coords: [row, col] position in the subplot grid, defaults to [0, 0].
    :type plot_coords: list[int], optional
    :param vmin: Minimum value for manual scaling, defaults to None.
    :type vmin: float, optional
    :param vmax: Maximum value for manual scaling, defaults to None.
    :type vmax: float, optional
    :param \**kwargs:
        * **title** (str): Custom title for the plot. Defaults to `symbol`.
        * **id** (str): Unique ID to distinguish instances of the same field nature.
        * **scale** (str): Scaling type, e.g., 'linear' or 'log'.
        * **ref_function** (callable): Analytical function for comparison.
    """

    timeline_instances = count(1)

    def __init__(self, key, symbol, plot_coords=[0, 0], vmin=None, vmax=None, **kwargs):
        self.key = key
        self.symbol = symbol
        self.plot_coords = plot_coords
        self.bounds = [vmin, vmax]
        LOG(self.bounds)

        self.title = kwargs.get("title", symbol)
        self.id = kwargs.get(
            "id", None
        )  # some custom id, to distinguish different instances of the same field nature (for example tau)
        self.scale = kwargs.get("scale", "linear")

        self.ref_function = kwargs.get("ref_function", None)
        self.pointsRef = []
        self.valuesRef = []

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_ref_data(self, points, values):
        self.pointsRef = points
        self.valuesRef = values

    def set_data(self, points, values):
        self.points = points
        self.values = values


class MapMovie2D(Data):
    r"""
    2D spatial field :math:`f(x, z, t)` rendered as a heatmap (pcolormesh) animation.
    """

    def __init__(
        self,
        key,
        symbol,
        plot_coords=[0, 0],
        cmap=DEFAULT_CMAP,
        norm="linear",
        streamlines=None,
        **kwargs,
    ):
        r"""
        Initializes a 2D movie field.

         (Refer to :class:`Data` for base parameters)
        :param cmap: Matplotlib colormap name, defaults to DEFAULT_CMAP.
        :type cmap: str, optional
        :param norm: Colorbar scaling. Options usually include 'linear', 'log', or 'TwoSlopeNorm'.
                     Defaults to "linear".
        :type norm: str, optional
        :param streamlines: A list of two Idefix field keys used to show vector streamlines,
                            e.g., ``["VX1", "VX2"]``. Defaults to None.
        :type streamlines: list[str], optional
        :param \**kwargs: Additional rendering options.
            :keyword streamline_color (str): Color of streamline arrows. Defaults to "w".
            :keyword compute (callable): Custom function to calculate new fields on the fly.
            :keyword contours (str): Field key used to draw contour lines over the pcolormesh.
            :keyword contour_color (str): Color of the contour lines. Defaults to "green".
        """

        # streamlines should be a list like ["VX1", "VX2"]

        super().__init__(key, symbol, plot_coords, **kwargs)
        self.cmap = cmap
        self.norm = norm
        self.streamlines = streamlines
        self.streamline_color = kwargs.get("streamline_color", "w")
        self.compute = kwargs.get("compute", None)
        self.contours = kwargs.get("contours", None)
        self.contour_color = kwargs.get("contour_color", "green")

    def set_norm(self, norm):
        self.norm = norm

    def set_cmap(self, cmap):
        self.cmap = cmap

    def set_XYgrid(self, X, Y):
        """
        Assign the spatial cartesian grid used for rendering the 2D pcolormesh.

        :param X: 2D array of horizontal coordinates.
        :type X: numpy.ndarray
        :param Y: 2D array of vertical coordinates.
        :type Y: numpy.ndarray
        """
        self.X, self.Y = X, Y


class Field1D(Data):
    """
    Base class for 1D fields :math:`f(x, t)`.
    Increments a global counter for indexing in results arrays.
    """

    def __init__(self, *args, **kwargs):
        self.index = next(Data.timeline_instances)
        super().__init__(*args, **kwargs)


class LineMovie1D(Field1D):
    """
    For :math:`f(x, t)` fields, renders as a line plot :math:`f(x, t)` that updates every frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpaceTimeHeatmap(Field1D):
    """
    For :math:`f(x, t)` fields, renders a space-time heatmap.

    :keyword cmap: Colormap for the heatmap.
    :keyword trace_over: List of :class:`PartQuantity` objects to overlay as trajectories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmap = kwargs.get("cmap", DEFAULT_CMAP)
        self.trace_over = kwargs.get("trace_over", [])


class PartQuantity(Data):
    """
    Tracks Lagrangian particle properties over time.
    """

    partQuantities_instances = count(1)

    def __init__(self, *args, **kwargs):
        self.index = next(PartQuantity.partQuantities_instances)
        super().__init__(*args, **kwargs)
