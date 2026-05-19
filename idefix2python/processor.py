from .vtk_io import readVTK
from . import tools
import numpy as np
from .quantities import PartQuantity, LineMovie1D, MapMovie2D

CARTESIAN_DIMENSION_NAMES = {
    "cartesian": [r"$x$", r"$y$", r"$z$"],
    "polar": [r"$x$", r"$y$", r"$z$"],
    "cylindrical": [r"$x$", r"$z$", None],
    "spherical": [r"$x$", r"$z$", r"$y$"],
}

DIMENSION_NAMES = {
    "cartesian": [r"$x$", r"$y$", r"$z$"],
    "polar": [r"$r$", r"$\phi$", r"$z$"],
    "cylindrical": [r"$r$", r"$z$", None],
    "spherical": [r"$r$", r"$\theta$", r"$\phi$"],
}


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

    def process(self, datavtk=None, partvtk=None):
        """
        Preprocessing data.
        - Build a common data structure to hold both datavtk and partvtk
        - Transposes and squeezes the vtk datas
        """
        if datavtk is not None:
            commonvtk = datavtk
            for qt in datavtk.data:
                if self.context.dimensions == 2:
                    datavtk.data[qt] = np.transpose(np.squeeze(datavtk.data[qt]))
                    datavtk.data[qt] = np.where(
                        self.gridInfo.mask, datavtk.data[qt], np.nan
                    )

                elif (
                    self.context.dimensions == 1
                    and len(np.shape(datavtk.data[qt])) == 3
                ):
                    datavtk.data[qt] = np.squeeze(datavtk.data[qt])
        else:
            commonvtk = partvtk

        if partvtk is not None:
            # the positions of the particles are in partvtk.x
            # Let's instead write that in datavtk
            commonvtk.data["PART_X1"] = tools.get_Position(
                partvtk, self.context.geometry, 0
            )
            commonvtk.data["PART_X2"] = tools.get_Position(
                partvtk, self.context.geometry, 1
            )
            commonvtk.data["PART_X3"] = tools.get_Position(
                partvtk, self.context.geometry, 2
            )
            # Let's also rename VXn to PART_VXn
            commonvtk.data["PART_VX1"] = partvtk.data.pop("VX1")
            commonvtk.data["PART_VX2"] = partvtk.data.pop("VX2")
            commonvtk.data["PART_VX3"] = partvtk.data.pop("VX3")

            # Let's write everything else to datavtk
            if datavtk is not None:
                for key in partvtk.data:
                    if key not in ["VX1", "VX2", "VX3"]:
                        if key in commonvtk.data:  # commonvtk is necessarly datavtk
                            raise Exception("processor is about to overwrite datavtk")
                        commonvtk.data[key] = partvtk.data[key]

        ## Custom computing. Now everything is stored in commonvtk
        for qtyInfo in self.qty_tocompute:
            commonvtk.data[qtyInfo.key] = np.squeeze(
                qtyInfo.compute(commonvtk)
            )  # is squeeze a good idea?

            # TODO safeguard for computed shape. Turns out to be not very straightforward.
            # computed_shape = np.shape(datavtk.data[qtyInfo.key])
            # if isinstance(qtyInfo, MapMovie2D) or isinstance(qtyInfo, LineMovie1D):
            #     expected_shape = self.gridInfo.shape
            # elif isinstance(qtyInfo, PartQuantity):
            #     expected_shape = np.shape(partvtk.data["uid"])
            # elif isinstance(qtyInfo, SpaceTimeHeatmap)
            # else:
            #     expected_shape = None
        return commonvtk

    def gather_1Cquantities(self, dataPath, partPath, quantities_togather):
        """
        quantities_togather must be single component particle. They can depend one variable, or one variable + time.
        """
        datavtk = None if dataPath is None else readVTK(dataPath)
        partvtk = None if partPath is None else readVTK(partPath)
        both_vtk_present = datavtk is not None and partvtk is not None

        if both_vtk_present and (datavtk.t - partvtk.t) > 1e-9:
            raise Exception(f"{dataPath} and {partPath} don't have the same time.")

        self.process(datavtk, partvtk)  # everything is now in datavtk
        gathered_1Cdata = [None] * (1 + len(quantities_togather))
        gathered_1Cdata[0] = datavtk.t[0]

        for qtyInfo in quantities_togather:
            key = qtyInfo.key
            gathering_index = qtyInfo.gathering_index
            if isinstance(qtyInfo, PartQuantity):
                gathered_1Cdata[gathering_index] = np.full(
                    self.context.particles_nb, np.nan
                )
                for ii, uid in enumerate(partvtk.data["uid"]):
                    gathered_1Cdata[gathering_index][uid] = partvtk.data[key][ii]

            else:
                gathered_1Cdata[gathering_index] = datavtk.data[key]

        return gathered_1Cdata


class GridInfo:
    def __init__(self, context, zoom=None):
        self.context = context
        self.geometry = context.geometry
        self.dimensions = context.dimensions
        self.grid_name_1, self.grid_name_2 = self.get_cartesian_grid_labels()
        self.axis_name_1, self.axis_name_2 = self.get_native_grid_labels()
        self.shape = None
        if self.context.outputTypes_info["vtk"].status:
            self.X1Line, self.X2Line = self.get_grid_line_points()
            if self.context.dimensions == 1:
                self.xmin = np.min(self.X1Line)
                self.xmax = np.max(self.X1Line)
                self.shape = np.shape(self.X1Line)
            elif self.context.dimensions == 2:
                # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
                self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)
                self.grid1, self.grid2 = tools.convertGrid_toXZ(
                    self.X1, self.X2, self.context.geometry
                )

                if not zoom:
                    self.mask = np.full(self.grid1.shape, True, dtype=bool)
                    self.mask = self.grid2 > 0
                    # )  # TODO hard coded, will be removed in later PR
                else:
                    self.mask = (
                        (self.grid1 < zoom) & (np.abs(self.grid2) < zoom)
                        # & (np.abs(np.pi / 2 - self.Theta) > np.pi / 12)
                    )
                self.xmin = 0  # works good atm
                self.xmax = np.max(np.where(self.mask, self.grid1, 0))
                self.ymax = np.max(np.where(self.mask, self.grid2, 0))
                self.ymin = np.min(np.where(self.mask, self.grid2, 0))

                self.shape = np.shape(self.X1)

    def get_cartesian_grid_labels(self):
        # 2D fields are always showed in cartesian. Thus, the labels should be cartesian.
        names = [None, None]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = CARTESIAN_DIMENSION_NAMES[self.context.geometry][dir]

        return names

    def get_native_grid_labels(self):
        names = [None, None]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = DIMENSION_NAMES[self.context.geometry][dir]
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
