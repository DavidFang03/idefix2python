from .vtk_io import readVTK
from . import tools
import numpy as np
from .quantities import PartQuantity, LocalQuantity


def pos_to_gridTSC(gridInfo, commonvtk):
    pX1 = commonvtk.data["PART_X1"]
    pX2 = commonvtk.data["PART_X2"]

    def get_nonuniform_tsc(pos_array, x, xl, dx):
        ci = np.searchsorted(xl, pos_array) - 1
        ci = np.clip(ci, 1, len(x) - 2)

        # d = (x - xl) / dx
        d = (pos_array - xl[ci]) / dx[ci]

        w_left = 0.5 * (1.0 - d) ** 2
        w_center = 0.75 - (d - 0.5) ** 2
        w_right = 0.5 * d**2

        return ci, (w_left, w_center, w_right)

    ci, weights_x1 = get_nonuniform_tsc(
        pX1, gridInfo.X1Line, gridInfo.X1LineL, gridInfo.dX1
    )

    if len(gridInfo.X2Line) > 1:
        cj, weights_x2 = get_nonuniform_tsc(
            pX2, gridInfo.X2Line, gridInfo.X2LineL, gridInfo.dX2
        )
        return (ci, weights_x1), (cj, weights_x2)
    else:
        return (ci, weights_x1), None


class PhysicsProcessor:
    def __init__(self, context, userArgs, streamLines=None):
        self.context = context
        self.userArgs = userArgs
        self.streamLines = streamLines

    def set_waitlist(self, waitlist):
        self.waitlist = waitlist

    def set_localQuantities(self, localQuantities):
        self.localQuantities = localQuantities

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
        for qtyInfo in self.waitlist:
            if isinstance(qtyInfo, LocalQuantity) and partvtk is not None:
                tsc_x1, tsc_x2 = pos_to_gridTSC(self.context.gridInfo, commonvtk)

                ci, (w1_l, w1_c, w1_r) = tsc_x1
                fluid_field = commonvtk.data[qtyInfo.localkey]

                if tsc_x2 is not None:
                    cj, (w2_l, w2_c, w2_r) = tsc_x2
                    interpolated_values = np.zeros(len(ci))

                    for dj, wj in zip([-1, 0, 1], [w2_l, w2_c, w2_r]):
                        for di, wi in zip([-1, 0, 1], [w1_l, w1_c, w1_r]):
                            fluid_vals = fluid_field[cj + dj, ci + di]
                            interpolated_values += (wj * wi) * fluid_vals

                    commonvtk.data[qtyInfo.key] = interpolated_values

                else:
                    interpolated_values = np.zeros(len(ci))
                    for di, wi in zip([-1, 0, 1], [w1_l, w1_c, w1_r]):
                        interpolated_values += wi * fluid_field[ci + di]

                    commonvtk.data[qtyInfo.key] = interpolated_values
            elif (
                (not isinstance(qtyInfo, PartQuantity) or partvtk is not None)
                and qtyInfo.compute is not None
            ):  # in the renderer there is no need to compute the partquantities again as they are already gathered
                commonvtk.data[qtyInfo.key] = qtyInfo.compute(commonvtk)
                # do not squeeze

            # TODO safeguard for computed shape. Turns out to be not very straightforward.
            # computed_shape = np.shape(datavtk.data[qtyInfo.key])
            # if isinstance(qtyInfo, MapMovie2D) or isinstance(qtyInfo, LineMovie1D):
            #     expected_shape = self.gridInfo.shape
            # elif isinstance(qtyInfo, PartQuantity):
            #     expected_shape = np.shape(partvtk.data["uid"])
            # elif isinstance(qtyInfo, SpaceTimeHeatmap)
            # else:
            #     expected_shape = None
            ## Local quantities

        return commonvtk

    def gather_1Cquantities(
        self, dataPath, partPath, quantities_togather, keys_tobound
    ):
        """
        quantities_togather must be single component particle. They can depend one variable, or one variable + time.
        """
        datavtk = None if dataPath is None else readVTK(dataPath)
        partvtk = None if partPath is None else readVTK(partPath)
        both_vtk_present = datavtk is not None and partvtk is not None

        if both_vtk_present and (datavtk.t[0] - partvtk.t[0]) > 1e-9:
            raise Exception(f"{dataPath} and {partPath} don't have the same time.")

        commonvtk = self.process(datavtk, partvtk)  # everything is now in datavtk
        gathered_1Cdata = {}
        gathered_1Cdata["TIME"] = commonvtk.t[0]

        for qtyInfo in quantities_togather:
            key = qtyInfo.key
            if isinstance(qtyInfo, PartQuantity) or isinstance(qtyInfo, LocalQuantity):
                gathered_1Cdata[key] = np.full(self.context.particles_nb, np.nan)
                for ii, uid in enumerate(commonvtk.data["uid"]):
                    gathered_1Cdata[key][uid] = commonvtk.data[key][ii]

            else:
                gathered_1Cdata[key] = commonvtk.data[key]

        # bounds
        bounds = {}
        for key in keys_tobound:
            bounds[key] = [
                np.nanmin(commonvtk.data[key]),
                np.nanmax(commonvtk.data[key]),
            ]

        return gathered_1Cdata, bounds


class PartsInfo:
    def __init__(self, active_directions):
        self.global_partsqty_togather = []
        X_index = active_directions[0]
        self.parts_X1 = PartQuantity(f"PART_X{X_index + 1}", uids="all")
        self.parts_X1.is_global = True
        self.global_partsqty_togather.append(self.parts_X1)

        if len(active_directions) >= 2:
            Y_index = active_directions[1]
            self.parts_X2 = PartQuantity(f"PART_X{Y_index + 1}", uids="all")

            self.parts_X2.is_global = True
            self.global_partsqty_togather.append(self.parts_X2)

        self.parts_Z = PartQuantity("PART_Z")
        self.parts_Z.is_global = True
