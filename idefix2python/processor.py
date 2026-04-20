from .vtk_io import readVTK
from . import tools
import numpy as np


class PhysicsProcessor:
    def __init__(self, context, userArgs, streamLines=None):
        self.context = context
        self.userArgs = userArgs
        self.streamLines = streamLines

        self._setup()

    def _setup(self):
        vtkInfo = self.context.outputTypes_info["vtk"]
        if vtkInfo.status:
            vtk = vtkInfo.vtk
            if vtkInfo.dimensions == 1:
                for direction in range(3):
                    try:
                        pos_array = tools.get_Position(
                            vtk, self.context.geometry, direction
                        )
                        # Check if pos_array is not None (to avoid crashing on cylindrical z)
                        if pos_array is not None and len(pos_array) > 1:
                            self.active_axis = direction
                            self.X1Line = pos_array
                            break
                    except (IndexError, TypeError):
                        continue
                self.axis_name = tools.get_Position_name(
                    self.context.geometry, self.active_axis
                )
                self.xmin = np.min(self.X1Line)
                self.xmax = np.max(self.X1Line)
            elif vtkInfo.dimensions == 2:
                # Find the two viable axes
                active_axes = []
                positions = []
                for direction in range(3):
                    try:
                        pos_array = tools.get_Position(
                            vtk, self.context.geometry, direction
                        )
                        if pos_array is not None and len(pos_array) > 1:
                            active_axes.append(direction)
                            positions.append(pos_array)
                    except (IndexError, TypeError):
                        continue

                if len(active_axes) != 2:
                    raise ValueError(
                        f"Expected exactly 2 active axes, found {len(active_axes)}: {active_axes}"
                    )
                self.axis1, self.axis2 = active_axes[0], active_axes[1]
                self.X1Line, self.X2Line = positions[0], positions[1]

                # Only (x,z) supported right now.
                self.axis_name_1 = r"$x$"
                self.axis_name_2 = r"$z$"

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

        # TODOs: support for slice1 and analysis and SPACETIMEHEATMAPS

    def set_fields(self, movies1D, movies2D):
        self.movies1D = movies1D
        self.movies2D = movies2D

    def process(self, V):
        """
        Transposes the vtk datas and add some stuff:
            - Soundspeed
            - Mach number
            - Positions of particles if there are
        Also collect the SpaceTimeHeatmaps
        """

        for qt in V.data:
            if self.context.dimensions == 2:
                V.data[qt] = np.transpose(V.data[qt][:, :, 0])
                V.data[qt] = np.where(self.mask, V.data[qt], np.nan)

            elif self.context.dimensions == 1 and len(np.shape(V.data[qt])) == 3:
                V.data[qt] = np.squeeze(V.data[qt])

        if "mass" in V.data:
            V.data["PART_X1"] = tools.get_Position(V, self.context.geometry, 0)
            V.data["PART_X2"] = tools.get_Position(V, self.context.geometry, 1)
            V.data["PART_X3"] = tools.get_Position(V, self.context.geometry, 2)

        for movie_dict in [self.movies1D, self.movies2D]:
            for key, qtyInfo in movie_dict.items():
                if hasattr(qtyInfo, "compute") and qtyInfo.compute is not None:
                    # Execute the user function.
                    # We pass the whole V.data so they can use multiple variables
                    # (e.g., Mach Number = velocity / sound_speed)
                    V.data[key] = qtyInfo.compute(V.data)

    def get_quantities(self, vtkPath, quantities):
        """
        quantities can be Quantities of Fields (can't find a better name...)
        """
        V = readVTK(vtkPath)
        self.process(V)
        PostSpaceTimeHeatmaps = [None] * (1 + len(quantities))
        PostSpaceTimeHeatmaps[0] = V.t[0]
        for key, field in quantities.items():
            PostSpaceTimeHeatmaps[field.index] = V.data[key]
        return PostSpaceTimeHeatmaps
