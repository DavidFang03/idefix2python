from .vtk_io import readVTK
from . import tools
import numpy as np
from .quantities import PartQuantity


class PhysicsProcessor:
    def __init__(self, context, userArgs, streamLines=None):
        self.context = context
        self.userArgs = userArgs
        self.streamLines = streamLines

        self._setup()

    def _setup(self):
        if self.context.outputTypes_info["vtk"].status:
            vtk = self.context.outputTypes_info["vtk"].vtk
            self.X1Line = tools.get_Position(
                vtk, self.context.geometry, self.context.active_directions[0]
            )
            self.axis_name_1 = self.context.active_directions_labels[0]
            if self.context.dimensions == 1:
                self.xmin = np.min(self.X1Line)
                self.xmax = np.max(self.X1Line)
            elif self.context.dimensions == 2:
                self.X2Line = tools.get_Position(
                    vtk, self.context.geometry, self.context.active_directions[1]
                )

                # 2D fields are always showed in cartesian. Thus, the labels should be cartesian.
                self.grid_name_1 = tools.get_Position_name_cartesian_equivalent(
                    self.context.geometry, self.context.active_directions[0]
                )
                self.grid_name_2 = tools.get_Position_name_cartesian_equivalent(
                    self.context.geometry, self.context.active_directions[1]
                )

                # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
                self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)

                self.grid1, self.grid2 = tools.convertGrid_toXZ(
                    self.X1, self.X2, self.context.geometry
                )

                if not self.userArgs.zoom:
                    self.mask = np.full(self.grid1.shape, True, dtype=bool)
                else:
                    self.mask = (
                        (self.grid1 < self.userArgs.zoom)
                        & (np.abs(self.grid2) < self.userArgs.zoom)
                        # & (np.abs(np.pi / 2 - self.Theta) > np.pi / 12)
                    )

                self.xmin = 0
                self.xmax = np.max(np.where(self.mask, self.grid1, 0))
                self.ymax = np.max(np.where(self.mask, self.grid2, 0))
                self.ymin = np.min(np.where(self.mask, self.grid2, 0))
        else:
            pass
            # TODO: support for particles

    def set_fields(self, movies1D, movies2D, partQuantities):
        self.movies1D = movies1D
        self.movies2D = movies2D
        self.partQuantities = partQuantities

    def set_times(self, times):
        self.times = times

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
                    V.data[qt] = np.where(self.mask, V.data[qt], np.nan)

                elif self.context.dimensions == 1 and len(np.shape(V.data[qt])) == 3:
                    V.data[qt] = np.squeeze(V.data[qt])

            for qtyInfo in [*self.movies1D, *self.movies2D]:
                if hasattr(qtyInfo, "compute") and qtyInfo.compute is not None:
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
