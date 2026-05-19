from itertools import count


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
    :param kwargs:
        * **title** (str): Custom title for the plot. Defaults to `symbol`.
        * **id** (str): Unique ID to distinguish instances of the same field nature.
        * **xmin** (float): Minimum x-axis bound.
        * **xmax** (float): Maximum x-axis bound.
        * **ymin** (float): Minimum y-axis bound.
        * **ymax** (float): Maximum y-axis bound.
        * **xscale** (str): X-axis scaling type, e.g., 'linear' or 'log'.
        * **yscale** (str): Y-axis scaling type, e.g., 'linear' or 'log'.
        * **style_kwargs** (dict): Style options forwarded to plotting calls.
        * **parts_kwargs** (dict): Style options forwarded to particles plotting calls.
        * **ref_function** (callable): Analytical function for comparison.
        * **compute** (callable): Custom function to calculate new fields on the fly.
    """

    def __init__(
        self, key, symbol="", plot_coords=[0, 0], vmin=None, vmax=None, **kwargs
    ):
        self.key = key
        self.symbol = symbol
        self.plot_coords = plot_coords
        self.bounds = [vmin, vmax]

        self.title = kwargs.get(
            "title", None
        )  # if None, will be replaced by symbol in ax
        self.id = kwargs.get(
            "id", None
        )  # some custom id, to distinguish different instances of the same field nature (for example tau)

        self.xmin = kwargs.get("xmin", None)
        self.xmax = kwargs.get("xmax", None)
        self.ymin = kwargs.get("ymin", None)
        self.ymax = kwargs.get("ymax", None)

        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        # heatmaps have a `norm` attribute

        self.style_kwargs = kwargs.get("style_kwargs", {})
        self.parts_kwargs = kwargs.get("parts_kwargs", {})

        self.ref_function = kwargs.get("ref_function", None)
        self.pointsRef = []
        self.valuesRef = []

        self.compute = kwargs.get("compute", None)

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_ref_data(self, points, values):
        self.pointsRef = points
        self.valuesRef = values

    def set_data(self, points, values):
        self.points = points
        self.values = values

    def set_norm(self, norm):
        self.norm = norm
        supported_norms = ["linear", "log", "TwoSlopeNorm"]
        if norm not in supported_norms:
            raise Exception(
                f"{norm} not implemented. Supported norms: {supported_norms}"
            )

    def __str__(self):
        return self.key


class MapMovie2D(Data):
    r"""
    2D spatial field :math:`f(x, z, t)` rendered as a heatmap (pcolormesh) animation.
    """

    def __init__(
        self,
        key,
        symbol="",
        plot_coords=[0, 0],
        norm="linear",
        streamlines=None,
        uids=None,
        **kwargs,
    ):
        r"""
        Initializes a 2D movie field.

         (Refer to :class:`Data` for base parameters)
        :param norm: Colorbar scaling. Options usually include 'linear', 'log', or 'TwoSlopeNorm'.
                     Defaults to "linear".
        :type norm: str, optional
        :param streamlines: A list of two Idefix field keys used to show vector streamlines,
                            e.g., ``["VX1", "VX2"]``. Defaults to None.
        :type streamlines: list[str], optional
        :param uids: List of the particles uid. Their trajectories will be showed over the maps. To show every particle, set it to "all".
                            e.g., ``[1,2]``. Defaults to None.
        :type uids: list[int] | Literal["all"] | None, optional
        :param \**kwargs: Additional rendering options.
            :keyword streamline_color (str): Color of streamline arrows. Defaults to "w".
            :keyword contours (Sequence[float] | None): Contour levels used to draw contour lines over the pcolormesh for this field. Defaults to None.
            :keyword contour_color (str): Color of the contour lines. Defaults to "green".
        """

        # streamlines should be a list like ["VX1", "VX2"]

        super().__init__(key, symbol, plot_coords, **kwargs)
        self.set_norm(norm)
        self.streamlines = streamlines
        self.streamline_color = kwargs.get("streamline_color", (1, 1, 1, 0.5))
        self.contours = kwargs.get("contours", None)
        self.contour_color = kwargs.get("contour_color", "green")
        self.uids = uids

    def set_XYgrid(self, X, Y):
        """
        Assign the spatial cartesian grid used for rendering the 2D pcolormesh.

        :param X: 2D array of horizontal coordinates.
        :type X: numpy.ndarray
        :param Y: 2D array of vertical coordinates.
        :type Y: numpy.ndarray
        """
        self.X, self.Y = X, Y

    def set_particles_trajectories(self, data):
        pass


class Field1D(Data):
    """
    Base class for 1D fields :math:`f(x, t)`.
    Increments a global counter for indexing in results arrays.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LineMovie1D(Field1D):
    """
    For :math:`f(x, t)` fields, renders as a line plot :math:`f(x, t)` that updates every frame.
    """

    def __init__(
        self,
        key,
        symbol="",
        plot_coords=[0, 0],
        vmin=None,
        vmax=None,
        uids=None,
        **kwargs,
    ):
        super().__init__(key, symbol, plot_coords, vmin, vmax, **kwargs)
        self.uids = uids
        self.is_movie = True
        self.is_timeline = False


class SpaceTimeHeatmap(Field1D):
    """
    For :math:`f(x, t)` fields, renders a space-time heatmap.

    :keyword cmap: Colormap for the heatmap.
    :keyword uids: List of particles' uids which trajectories will be displayed.
    """

    instances = count(1)

    def __init__(
        self,
        key,
        symbol="",
        plot_coords=[0, 0],
        vmin=None,
        vmax=None,
        norm="linear",
        uids=None,
        **kwargs,
    ):
        super().__init__(key, symbol, plot_coords, vmin, vmax, **kwargs)
        self.set_norm(norm)
        self.uids = uids
        self.index = next(SpaceTimeHeatmap.instances)
        self.is_timeline = True
        self.is_movie = False


class OneComponentOneVariable(Data):
    """
    A y(x) value where x can be any variable. If xqty is None, that means x is time and the quantity will be treated as a timeline.
    Otherwise, it will be treated as a LineMovie1D.

    """

    _key_index_map = {}

    def __init__(
        self,
        key,
        symbol="",
        plot_coords=[0, 0],
        vmin=None,
        vmax=None,
        xqty=None,
        **kwargs,
    ):
        if key not in OneComponentOneVariable._key_index_map:
            OneComponentOneVariable._key_index_map[key] = (
                len(OneComponentOneVariable._key_index_map) + 1
            )
        self.index = OneComponentOneVariable._key_index_map[key]
        super().__init__(key, symbol, plot_coords, vmin, vmax, **kwargs)
        if kwargs.get("uids", None) is not None:
            raise Exception(
                "For uid specific 1C1V quantity, please use PartQuantity instead."
            )

        self.xqty = xqty  # if None, it will be time.
        self.is_timeline = xqty is None
        self.is_movie = xqty is not None


class PartQuantity(Data):
    """
    Particular case of OneComponentOneVariable when the variable is time and that there is one value per particle (so not really one component but rather Npart components...)
    Tracks Lagrangian particle properties over time.

    :keyword: uids (optional) the ids of the particles wanted.
        Defaults to "all" (all particles)
    """

    _key_index_map = {}

    def __init__(
        self,
        key,
        symbol="",
        plot_coords=[0, 0],
        vmin=None,
        vmax=None,
        uids="all",
        **kwargs,
    ):
        if key not in PartQuantity._key_index_map:
            PartQuantity._key_index_map[key] = len(PartQuantity._key_index_map) + 1
        self.index = PartQuantity._key_index_map[key]
        super().__init__(key, symbol, plot_coords, vmin, vmax, **kwargs)
        self.uids = uids
        self.is_global = False  # default
        self.is_timeline = True
        self.is_movie = False
