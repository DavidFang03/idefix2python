from .vtk_io import readVTK
from . import tools
import numpy as np
from .quantities import PartQuantity, LineMovie1D, MapMovie2D


class PhysicsProcessor:
    def __init__(self, context, userArgs, streamLines=None):
        self.context = context
        self.userArgs = userArgs
        self.streamLines = streamLines

        self.gridInfo = GridInfo(self.context, self.userArgs.zoom)

    def set_qty_tocompute(self, qty_tocompute):
        self.qty_tocompute = qty_tocompute

    def set_partQuantities(self, partQuantities):
        self.partQuantities = partQuantities

    def set_vtktimes(self, vtktimes):
        self.vtktimes = vtktimes

    def process(self, V):
        """
        Transposes the vtk datas and add some stuff:
            - Soundspeed
            - Mach number
            - Positions of particles if there are
        Also collect the SpaceTimeHeatmaps
        """
        is_particle_vtk = "uid" in V.data
        if not is_particle_vtk:
            for qt in V.data:
                if self.context.dimensions == 2:
                    V.data[qt] = np.transpose(np.squeeze(V.data[qt]))
                    V.data[qt] = np.where(self.gridInfo.mask, V.data[qt], np.nan)

                elif self.context.dimensions == 1 and len(np.shape(V.data[qt])) == 3:
                    V.data[qt] = np.squeeze(V.data[qt])

            for qtyInfo in self.qty_tocompute:
                if isinstance(qtyInfo, MapMovie2D) or isinstance(
                    qtyInfo, LineMovie1D
                ):  # partQuantities and spacetimesheatmap are not concerned by this.
                    V.data[qtyInfo.key] = qtyInfo.compute(
                        V.data
                    )  # TODO Add safeguard for computed shape

        else:
            V.data["PART_X1"] = tools.get_Position(V, self.context.geometry, 0)
            V.data["PART_X2"] = tools.get_Position(V, self.context.geometry, 1)
            V.data["PART_X3"] = tools.get_Position(V, self.context.geometry, 2)

            for qtyInfo in [*self.partQuantities]:
                if hasattr(qtyInfo, "compute") and qtyInfo.compute is not None:
                    # Currently the computed shape must be (len(V.data["uid"]))
                    computed_data = qtyInfo.compute(
                        V
                    )  # TODO Add safeguard for computed shape
                    if len(computed_data) != len(V.data["uid"]):
                        raise ValueError(
                            f"The computed data has shape {np.shape(computed_data)} but should have the same shape as V.data['uid']"
                        )
                    V.data[qtyInfo.key] = computed_data

    def get_quantities(self, vtkPath, quantities):
        """
        quantities can be Quantities of Fields (can't find a better name...)
        """
        V = readVTK(vtkPath)
        self.process(V)
        PostSpaceTimeHeatmaps = [None] * (1 + len(quantities))
        PostSpaceTimeHeatmaps[0] = V.t[0]
        for field in quantities:
            key = field.key
            if isinstance(field, PartQuantity):
                PostSpaceTimeHeatmaps[field.index] = np.full(
                    self.context.particles_nb, np.nan
                )
                for ii, uid in enumerate(V.data["uid"]):
                    PostSpaceTimeHeatmaps[field.index][uid] = V.data[key][ii]

            else:
                PostSpaceTimeHeatmaps[field.index] = V.data[key]

        return PostSpaceTimeHeatmaps


class GridInfo:
    def __init__(self, context, zoom=None):
        self.context = context
        self.geometry = context.geometry
        self.dimensions = context.dimensions
        self.grid_name_1, self.grid_name_2 = self.get_cartesian_grid_labels()
        self.axis_name_1, self.axis_name_2 = self.get_native_grid_labels()

        if self.context.outputTypes_info["vtk"].status:
            self.X1Line, self.X2Line = self.get_grid_line_points()
            if self.context.dimensions == 1:
                self.xmin = np.min(self.X1Line)
                self.xmax = np.max(self.X1Line)
            elif self.context.dimensions == 2:
                # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
                self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)
                self.grid1, self.grid2 = tools.convertGrid_toXZ(
                    self.X1, self.X2, self.context.geometry
                )

                if not zoom:
                    self.mask = np.full(self.grid1.shape, True, dtype=bool)
                    self.mask = (
                        self.grid2 >= 0
                    )  # TODO hard coded, will be removed in later PR
                else:
                    self.mask = (
                        (self.grid1 < zoom) & (np.abs(self.grid2) < zoom)
                        # & (np.abs(np.pi / 2 - self.Theta) > np.pi / 12)
                    )
                self.xmin = 0  # works good atm
                self.xmax = np.max(np.where(self.mask, self.grid1, 0))
                self.ymax = np.max(np.where(self.mask, self.grid2, 0))
                self.ymin = np.min(np.where(self.mask, self.grid2, 0))

    def get_cartesian_grid_labels(self):
        # 2D fields are always showed in cartesian. Thus, the labels should be cartesian.
        names = [None, None]

        match self.context.geometry:
            case "cartesian":
                positions = [r"$x$", r"$y$", r"$z$"]
            case "polar":
                positions = [r"$x$", r"$y$", r"$z$"]
            case "cylindrical":
                positions = [r"$x$", r"$z$", None]
            case "spherical":
                positions = [r"$x$", r"$z$", r"$y$"]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = positions[dir]

        return names

    def get_native_grid_labels(self):
        names = [None, None]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = dir
        return names

    def get_grid_line_points(self):
        Lines = [None, None]

        vtk = self.context.outputTypes_info["vtk"].vtk
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                Lines[i] = tools.get_Position(vtk, self.context.geometry, dir)

        return Lines


class PartsInfo:
    def __init__(self, active_directions):
        self.partsqty_togather = []
        X_index = active_directions[0]
        self.parts_X1 = PartQuantity(f"PART_X{X_index + 1}", uids="all")
        self.parts_X1.is_global = True
        self.partsqty_togather.append(self.parts_X1)

        if len(active_directions) >= 2:
            Y_index = active_directions[1]
            self.parts_X2 = PartQuantity(f"PART_X{Y_index + 1}", uids="all")

            self.parts_X2.is_global = True
            self.partsqty_togather.append(self.parts_X2)

        self.parts_Z = PartQuantity("PART_Z")
        self.parts_Z.is_global = True
