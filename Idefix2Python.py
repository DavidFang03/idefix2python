# sys.path.append(os.getenv("IDEFIX_DIR"))
# from pytools.vtk_io import readVTK
import glob
import json
import os

# import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tools
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import RegularGridInterpolator
from vtk_io import readVTK
import argparse
import register
import dust
from pathlib import Path
from itertools import count, repeat


plt.style.use("dark_background")
# plt.rcParams.update(
#     {
#         # "text.usetex": True,
#     }
# )
plt.rcParams["hatch.color"] = "gray"
plt.rcParams["hatch.linewidth"] = 0.5

DEFAULT_CMAP = "berlin"
DPI = 300
CPU = 1

# def averageTheta(R, Theta, Values):
#     """
#     R and Theta must be lines.
#     I assume log for R and uniform on theta (on the edges at least)
#     """
#     # dr = np.diff(R)  # log
#     # dr_r = dr / R[:-1]
#     # dr = np.append(dr, dr_r[-1] * R[-1])
#     dtheta = np.diff(Theta)  # lin on the edges
#     dtheta = np.append(dtheta, dtheta[-1])

#     Rmesh, dthetamesh = np.meshgrid(R, dtheta)

#     return np.sum(Values * Rmesh * dthetamesh, axis=1)


lw_streamline = 0.2
density_streamline = [1, 2]
arrowstyle_streamline = "->"


def LOG(txt):
    print(txt)


class RunContext:
    """
    The input. Also creates some frames directories.
    """

    def __init__(self, projectPath, runName, args, **kwargs):
        tools.RequirePath(projectPath, dir_or_file="dir")

        self.projectPath = Path(projectPath)
        self.runName = runName
        self.args = args

        configPath = kwargs.get("configPath", "./config.json")
        self.config = tools.process_configs(configPath)

        self.dataPath = self.projectPath / "outputs" / runName
        self.iniPath = self.projectPath / "inputs" / f"{runName}.ini"
        self.format_inputs_text = tools.formatInputs(self.iniPath)

        self._setup_directories()
        self._check_data()

    def _setup_directories(self):
        self.frameRootFolder = self.projectPath / "frames" / self.runName
        self.globalFolder = self.frameRootFolder / "global"
        self.slice1Folder = self.frameRootFolder / "slice1"
        self.videosFolder = self.projectPath / "videos"

        for path in [
            self.globalFolder,
            self.slice1Folder,
            self.videosFolder,
        ]:
            os.makedirs(path, exist_ok=True)
            # except OSError as _:
            # pass
            # subfolder = os.path.basename(path)
            # content = glob.glob(f"{path}/*")
            # user_agree = input(
            #     f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
            # )
            # if user_agree == "r":
            #     for f in content:
            #         os.remove(f)
            # elif user_agree == "n":
            #     exit()

    def _check_data(self):
        "Show fields in every kind of data and detect is there are Pressure, B, Dust or Particles fields. Also detects the geometry. Also detect t_start and t_end"
        self.outputTypes_info = {}
        # self.outputTypes = ["analysis", "slice1", "vtk", "particles"]
        self.outputTypes = ["slice1", "vtk", "particles"]
        # self.outputTypes_info["analysis"] = OutputType_info(self.analysis_path, "analysis")
        self.outputTypes_info["vtk"] = OutputType_info(
            "vtk", self.get_global_vtkFiles()
        )
        self.outputTypes_info["slice1"] = OutputType_info(
            "slice1", self.get_slice1_vtkFiles()
        )
        self.outputTypes_info["particles"] = OutputType_info(
            "particles", self.get_particles_vtkFiles()
        )
        self.outputTypes_info["particles"].dimensions = self.outputTypes_info[
            "vtk"
        ].dimensions
        # There's no way to deduce the number of dimensions from the part*.vtk files but it has to be the same as in the global vtk

        LOG(f"------ Available fields in {self.dataPath} ------")
        for outputType in self.outputTypes:
            outputTypeInfo = self.outputTypes_info[outputType]
            geometry = outputTypeInfo.geometry
            dimensions = outputTypeInfo.dimensions
            if geometry is not None:
                self.geometry = geometry
            if dimensions is not None:
                self.dimensions = outputTypeInfo.dimensions

    def get_global_vtkFiles(self, end=1):
        pattern = "vtks/data*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        return filelist[: int(len(filelist) * end)]

    def get_slice1_vtkFiles(self, end=1):
        pattern = "vtks/slice1*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        return filelist[: int(len(filelist) * end)]

    def get_particles_vtkFiles(self, end=1):
        pattern = "vtks/part*.vtk"
        # filelist = sorted(self.dataPath.glob(pattern))
        filelist = sorted(
            Path("/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup_l").glob(
                "part*.vtk"
            )
        )
        return filelist[: int(len(filelist) * end)]


class OutputType_info:
    """
    Different types of output: global (vtk), slice (vtk), timevol (dat), particles (vtk)
    """

    def __init__(self, name, files):
        self.name = name
        self.files = files
        self.geometry = None
        self.dimensions = None

        self.dataHas = {
            "Pressure": False,
            "B": False,
            "Dust": False,
            "Particles": False,
        }

        if len(self.files) > 0:
            self.status = True
            self.test_file = self.files[0]
            self.ext = self.test_file.suffixes[-1]

            self._set_testData()
            self.get_availableKeys()

        else:
            self.status = False

    def _set_testData(self):
        if "vtk" in self.ext:
            vtk = readVTK(self.test_file)
            self.testData = vtk.data
            self.geometry = vtk.geometry
            self.dimensions = vtk.dimensions
            self.vtk = vtk
        elif "dat" in self.ext:
            self.testData = tools.dat_to_dict(self.test_file)

    def get_availableKeys(self):
        if not self.status:
            return f"No {self.name} present"

        for qt in self.testData.keys():
            LOG(f"{qt:>10} {np.shape(self.testData[qt].data)}")
            if qt == "PRS":
                self.dataHas["Pressure"] = True
            elif qt == "BX1":
                self.dataHas["B"] = True
            elif qt == "Dust0_RHO":
                self.dataHas["Dust"] = True
            elif qt == "PART_VX1":
                print("CCCCCCCCCCCCCCCC")
                self.dataHas["Particles"] = True
        if "vtk" in self.ext:
            dataStart = self.files[0]
            dataEnd = self.files[-1]  # TODO end option
            self.tStart = readVTK(dataStart).t[0]
            self.tEnd = readVTK(dataEnd).t[0]
        elif "dat" in self.ext:
            raise NotImplementedError()
        return f"{self.name}:", self.testData.keys()

    def set_Times(self, Times):
        self.Times = Times


class Data:
    # Types of data that I can imagine:
    # 2D Field
    # 1D Field time graph
    # Scalar time graph
    timeline_instances = count(1)

    def __init__(self, key, symbol, plot_coords, **kwargs):
        self.key = key
        self.symbol = symbol
        self.plot_coords = plot_coords
        self.vmin, self.vmax = [None, None]

        self.title = kwargs.get("title", symbol)
        self.id = kwargs.get(
            "id", None
        )  # some custom id, to distinguish different instances of the same field nature (for example tau)
        self.scale = kwargs.get("scale", "linear")

    def set_bounds(self, vmin, vmax):
        self.bounds = [vmin, vmax]


class Field2D(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmap = kwargs.get("cmap", DEFAULT_CMAP)
        self.streamline = kwargs.get("streamline")

    def set_XYgrid(self, X, Y):
        self.X, self.Y = X, Y


class Field1D(Data):
    def __init__(self, *args, **kwargs):
        self.index = next(Data.timeline_instances)
        super().__init__(*args, **kwargs)
        self.cmap = kwargs.get("cmap", DEFAULT_CMAP)

        self.pointsRef = []
        self.valuesRef = []

    def set_data(self, points, values):
        self.points = points
        self.values = values

    def set_ref_data(self, points, values):
        """
        If one wants to add a fit
        """
        self.pointsRef = points
        self.valuesRef = values


class Quantity(Data):
    Quantities_instances = count(1)

    def __init__(self, *args, **kwargs):
        self.index = next(Data.timeline_instances)
        super().__init__(*args, **kwargs)


class PartQuantity(Data):
    partQuantities_instances = count(1)

    def __init__(self, *args, **kwargs):
        self.index = next(PartQuantity.partQuantities_instances)
        super().__init__(*args, **kwargs)

    def set_data(self, points, values):
        self.points = points
        self.values = values

    def set_ref_data(self, points, values):
        """
        If one wants to add a fit
        """
        self.pointsRef = points
        self.valuesRef = values


def get_streamplot_data(
    X1, X2, Ux_data, Uy_data, geometry, zoom=None, resolution=200, method="linear"
):
    """
    X,Y must be Lines, Ux and Uy are the data points.
    Returns the 1d arrays x_coords and y-coords, with the interpolated values Ux_uni, Uy_uni (2D arrays) that can go directly in streamplot
    """
    # Interpolate U in (X1, X2) system
    Ux_interp = RegularGridInterpolator(
        (X1, X2),
        Ux_data.T,
        fill_value=np.nan,
        method=method,
        bounds_error=False,
    )
    Uy_interp = RegularGridInterpolator(
        (X1, X2),
        Uy_data.T,
        fill_value=np.nan,
        method=method,
        bounds_error=False,
    )

    # Now U can be estimated for any value of (X1, X2)
    # However we need a uniform (X,Z) grid.
    # The bounds [xmin, xmax] and [zmin, zmax] depend on the geometry
    # Also the eventual zoom has to be taken account.

    if geometry in ["cylindric", "cartesian"]:
        x_min = np.min(X1)
        if zoom is None:
            x_max = np.max(X2)
            z_min, z_max = np.min(X2), np.max(X2)
        else:
            x_max = zoom
            z_min, z_max = -zoom, zoom

        x_coords = x_min + np.arange(resolution) * ((x_max - x_min) / (resolution - 1))
        z_coords = z_min + np.arange(resolution) * ((z_max - z_min) / (resolution - 1))

        X_uni, Z_uni = np.meshgrid(x_coords, z_coords)
        pts = np.stack((X_uni, Z_uni), axis=-1)

    elif geometry == "spherical":
        r_min, r_max = np.min(X1), np.max(X1)
        r_max = r_max if zoom is None else zoom
        z_min, z_max = -r_max, r_max
        x_coords = r_min + np.arange(resolution) * ((r_max - r_min) / (resolution - 1))
        z_coords = z_min + np.arange(resolution) * ((z_max - z_min) / (resolution - 1))

        X_uni, Z_uni = np.meshgrid(x_coords, z_coords)
        R_fromuni = np.sqrt(X_uni**2 + Z_uni**2)
        Theta_fromuni = np.pi / 2 - np.atan(Z_uni / X_uni)
        pts = np.stack((R_fromuni, Theta_fromuni), axis=-1)

    else:
        raise NotImplementedError("This geometry hasn't been implemented yet.")

    return (
        x_coords,
        z_coords,
        Ux_interp(pts),
        Uy_interp(pts),
    )


class StreamLinesData:
    def __init__(self, name, color, title=None):
        self.name = name
        self.title = name if title is None else title
        self.color = color

    def set_data(self, X, Z, dataX, dataZ):
        self.X = X
        self.Z = Z
        self.dataX = dataX
        self.dataZ = dataZ


def get_Position(file, geometry, direction):
    match geometry:
        case "cartesian":
            positions = [file.x, file.y, file.z]
        case "polar":
            positions = [file.x, file.y, file.z]
        case "cylindrial":
            positions = [file.r, file.z]
        case "spherical":
            positions = [file.x, file.theta, file.phi]
    return positions[direction]


class PhysicsProcessor:
    def __init__(self, context, userArgs):
        self.context = context
        self.userArgs = userArgs

        self._setup()

    def set_fields(self, fields1D, fields2D):
        self.fields1D = fields1D
        self.fields2D = fields2D

    def process(self, V):
        """
        Transposes the vtk datas and add some stuff:
            - Soundspeed
            - Mach number
            - Positions of particles if there are
        Also collect the Spatiograms
        """

        for qt in V.data:
            if self.context.dimensions == 2:
                V.data[qt] = np.transpose(V.data[qt][:, :, 0])
                V.data[qt] = np.where(self.mask, V.data[qt], np.nan)

            elif self.context.dimensions == 1 and len(np.shape(V.data)) == 3:
                print(np.shape(V.data[qt]))
                V.data[qt] = np.transpose(V.data[qt][:, 0, 0])

        if "PRS" in V.data:
            V.data["cs"] = tools.divide_discardingNullDenominator(
                np.sqrt(V.data["PRS"]), np.sqrt(V.data["RHO"])
            )
        if "cs" in V.data:
            V.data["Mach_p"] = tools.divide_discardingNullDenominator(
                np.sqrt(V.data["VX1"] ** 2 + V.data["VX2"] ** 2),
                V.data["cs"],
            )

        # For streamlines
        if self.context.dimensions == 2:
            V.data["vx"], V.data["vz"] = tools.convertVector_toXZ(
                V.data["VX1"],
                V.data["VX2"],
                self.X1,
                self.X2,
                self.geometry,
            )
            if "BX1" in V.data:
                V.data["Bx"], V.data["Bz"] = tools.convertVector_toXZ(
                    V.data["BX1"],
                    V.data["BX2"],
                    self.X1,
                    self.X2,
                    self.geometry,
                )
            if "Dust0_VX1" in V.data:
                V.data["vxDust"], V.data["vzDust"] = tools.convertVector_toXZ(
                    V.data["Dust0_VX1"],
                    V.data["Dust0_VX1"],
                    self.X1,
                    self.X2,
                    self.geometry,
                )
        print(V.data.keys())
        print("mass" in V.data)
        if "mass" in V.data:
            print("DDDDDDDDDDDDDD")
            V.data["PART_X1"] = get_Position(V, self.context.geometry, 0)
            V.data["PART_X2"] = get_Position(V, self.context.geometry, 1)
            V.data["PART_X3"] = get_Position(V, self.context.geometry, 2)

    def _setup(self):
        vtkInfo = self.context.outputTypes_info["vtk"]
        if vtkInfo.status:
            vtk = vtkInfo.vtk
            if vtkInfo.dimensions == 1:
                self.X1Line = get_Position(vtk, self.context.geometry, 0)
            elif vtkInfo.dimensions == 2:
                self.X1Line = get_Position(vtk, self.context.geometry, 0)
                self.X2Line = get_Position(vtk, self.context.geometry, 1)

                # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
                self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)

                self.X, self.Z = tools.convertGrid_toXZ(
                    self.X1, self.X2, self.context.geometry
                )

                if not self.userArgs.zoom:
                    self.mask = np.full(self.X.shape, True, dtype=bool)
                else:
                    self.mask = (
                        (self.X < self.userArgs.zoom)
                        & (np.abs(self.Z) < self.userArgs.zoom)
                        # & (np.abs(np.pi / 2 - self.Theta) > np.pi / 12)
                    )

                self.xmin = 0
                self.xmax = np.max(np.where(self.mask, self.X, 0))
                self.ymax = np.max(np.where(self.mask, self.Z, 0))
                self.ymin = np.min(np.where(self.mask, self.Z, 0))

        # TODOs: support for slice1 and analysis and SPATIOGRAMS

    def get_quantities(self, vtkPath, quantities):
        """
        quantities can be Quantities of Fields (can't find a better name...)
        """
        V = readVTK(vtkPath)
        self.process(V)
        PostSpatiograms = [None] * (1 + len(quantities))
        PostSpatiograms[0] = V.t[0]
        for key, field in quantities.items():
            PostSpatiograms[field.index] = V.data[key]
        return PostSpatiograms


class SliceRenderer:
    def __init__(
        self,
        context,
        processor,
        fields1D,
        fields2D,
        partQuantities,
        userArgs,
        framesPaths,
    ):
        self.context = context
        self.processor = processor
        self.fields2D = fields2D
        self.fields1D = fields1D
        self.partQuantities = partQuantities
        self.userArgs = userArgs
        self.framesPaths = framesPaths

        self.doStreamLines = True  # TODO should be moved in userArgs

    def render2D(self, vtkPath):
        V = readVTK(vtkPath)
        X = self.processor.X
        Z = self.processor.Z
        self.processor.process(V)
        time = V.t[0]

        if not self.userArgs.onlyAnalysis:
            rows = 3
            columns = (
                max([qtyInfo.plot_coords[1] for qtyInfo in self.fields2D.values()]) + 1
            )
            cbars = {}
            fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 16))
            if self.userArgs.zoom:
                fig.patch.set_linewidth(10)
                fig.patch.set_edgecolor("cornflowerblue")
            fig.subplots_adjust(
                left=0.05,
                right=1 - 0.05,
                bottom=0.07,
                top=0.83,
                hspace=0.25,
                wspace=0.02,
            )

            fig.suptitle(f"{self.name}\n{vtkPath}\n$t={time:.1e}$")

            tools.annotateInputs(axs, self.formatInputs_text)

            if self.doStreamLines:
                self.StreamLines["Vp"].set_data(
                    *get_streamplot_data(
                        self.processor.X1Line,
                        self.processor.X2Line,
                        V.data["vx"],
                        V.data["vz"],
                        self.context.geometry,
                        self.processor.xmax,
                    )
                )
                if self.context.dataHas["B"]:
                    self.StreamLines["Bp"].set_data(
                        *get_streamplot_data(
                            self.processor.X1Line,
                            self.processor.X2Line,
                            V.data["Bx"],
                            V.data["Bz"],
                            self.context.geometry,
                            self.processor.xmax,
                        )
                    )
                if self.context.dataHas["Dust"]:
                    self.StreamLines["VpDust"].set_data(
                        *get_streamplot_data(
                            self.processor.X1Line,
                            self.processor.X2Line,
                            V.data["vxDust"],
                            V.data["vzDust"],
                            self.processor.xmax,
                        )
                    )

            unusedAxs = [[i, j] for i in range(rows) for j in range(columns)]
            for qty, qtyInfo in self.fields2D.items():
                data = V.data[qty]
                ax = axs[*qtyInfo.plot_coords]
                unusedAxs.remove(qtyInfo.plot_coords)
                streamline = qtyInfo.streamline
                if None in qtyInfo.bounds or self.userArgs.noBounds:
                    vmin, vmax = np.nanmin(data), np.nanmax(data)
                else:
                    vmin, vmax = qtyInfo.bounds

                cbar_format = FuncFormatter(tools.fmt)
                norm = Normalize(vmin=vmin, vmax=vmax)

                if qtyInfo.norm == "log":
                    vmin = vmin if vmin > 0 else 1e-9
                    vmax = vmax if vmax > 0 else 1e-8
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                    cbar_format = None
                elif qtyInfo.norm == "TwoSlopeNorm" and not self.userArgs.noBounds:
                    vmin = vmin if vmin < 0 else -1e-7
                    vmax = vmax if vmax > 0 else 1e-7
                    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

                else:
                    vmin, vmax = qtyInfo.bounds

                cmesh = ax.pcolormesh(
                    X,
                    Z,
                    data,
                    cmap=qtyInfo.cmap,
                    norm=norm,
                )

                if self.userArgs.zoom:
                    ax.contourf(
                        X,
                        Z,
                        np.logical_not(self.mask),
                        levels=[0.5, 1],
                        hatches=["////"],
                        colors="none",
                    )

                cbar = fig.colorbar(cmesh, ax=ax, format=cbar_format)
                cbar.ax.set_title(qtyInfo.name)
                cbars[qty] = cbar
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$x$")
                ax.set_xlim(self.processor.xmin, self.processor.xmax)
                ax.set_ylim(self.processor.ymin, self.processor.ymax)

                if qtyInfo.plot_coords[1] == 0:
                    ax.set_ylabel(r"$z$")
                title = qtyInfo.name
                color = None

                if self.doStreamLines and streamline:
                    streamlineData = self.processor.StreamLines[streamline]
                    ax.streamplot(
                        streamlineData.X,
                        streamlineData.Z,
                        streamlineData.dataX,
                        streamlineData.dataZ,
                        density=density_streamline,
                        linewidth=lw_streamline,
                        arrowstyle=arrowstyle_streamline,
                        color=streamlineData.color,
                    )

                    title = rf"{streamlineData.title} $\nearrow$"
                    color = streamlineData.color

                ax.set_title(title, color=color)

            if self.context.dataHas["Pressure"]:
                Mach_pInfo = self.fields2D["Mach_p"]
                level0 = axs[*Mach_pInfo.coords].contour(
                    X,
                    Z,
                    V.data["Mach_p"],
                    [1],
                    alpha=0.5,
                    colors=["green"],
                    linewidths=[1.5],
                )
                cbarMach_p = cbars["Mach_p"]
                cbarMach_p.add_lines(level0)

            for coords in unusedAxs:
                axs[*coords].remove()

            slice1_name = vtkPath.name
            slice1_png_path = self.slice1_png_pattern.replace(
                "*", f"{slice1_name[:-4]}"
            )
            fig.savefig(slice1_png_path, dpi=300)
            plt.close(fig)
            LOG(f"[OK] {slice1_png_path}")

    def render1D(self, vtkPath):
        """
        Currently plotting everything on one plot
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.tight_layout()

        postSpatiograms = self.processor.get_quantities(vtkPath, self.fields1D)

        vtk = readVTK(vtkPath)
        R = vtk.x

        t = postSpatiograms[0]
        fig.suptitle(rf"$t={t:.1e}$")

        for key, field1D in self.fields1D.items():
            # key, field1D = "RHO", self.fields1D["RHO"]
            index = field1D.index
            ax.plot(R, postSpatiograms[index])
            # ax.plot(self.processor.X1Line, postSpatiograms[index])
            ax.set_xlabel(r"$R$")
            ax.set_ylabel(field1D.symbol)
            ax.set_title(field1D.title)
        # ax.set_xlim(1.45, 2.4)
        # ax.set_ylim(0, 0.01)
        # ax.set_ylim(np.min(postSpatiograms[4]), np.max(postSpatiograms[4]))
        ax.grid()

        slice1_name = vtkPath.name
        slice1_png_path = Path(
            str(self.framesPaths.slice1_png_pattern).replace("*", f"{slice1_name[:-4]}")
        )
        fig.savefig(slice1_png_path, dpi=300)
        plt.close(fig)
        LOG(f"[OK] {slice1_png_path}")

    def render_Spatiogram(self):
        rows, columns = 3, 3
        fig, axs = plt.subplots(rows, columns, figsize=(16, 12))
        fig.tight_layout()

        unusedAxs = [[i, j] for i in range(rows) for j in range(columns)]

        for key, field1D in self.fields1D.items():
            ax = axs[*field1D.plot_coords]
            unusedAxs.remove(field1D.plot_coords)
            T, Points = np.meshgrid(
                np.asarray(self.context.outputTypes_info["vtk"].Times),
                np.asarray(field1D.points),
            )
            cmesh = ax.pcolormesh(
                T,
                Points,
                np.transpose(field1D.values),
                shading="nearest",
                # TODO More flexible norm
                cmap="inferno",
            )
            cbar = fig.colorbar(cmesh, ax=ax)

            if len(field1D.pointsRef) > 0:
                ax.plot(field1D.pointsRef, field1D.valuesRef, label="Predicted")
                ax.legend()
            cbar.ax.set_title(field1D.symbol)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$R$")
            ax.set_title(field1D.title)
            ax.grid()

        for coords in unusedAxs:
            axs[*coords].remove()
        path = self.framesPaths.spatiogram_frame_path
        fig.savefig(path, dpi=DPI)
        LOG(f"[OK] {path}")

    def render_TimeSeries(self):
        rows, columns = 2, 2
        fig, axs = plt.subplots(rows, columns, figsize=(16, 12))
        fig.tight_layout()

        unusedAxs = [[i, j] for i in range(rows) for j in range(columns)]

        for key, qty in self.partQuantities.items():
            ax = axs[*qty.plot_coords]
            unusedAxs.remove(qty.plot_coords)
            T = np.asarray(self.context.outputTypes_info["particles"].Times)
            ax.plot(T, qty.values, lw=2)

            if len(qty.pointsRef) > 0:
                ax.plot(qty.pointsRef, qty.valuesRef, ls="--", lw=2, label="Predicted")
                ax.legend()

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$R$")
            ax.set_title(qty.title)
            ax.grid()

        for coords in unusedAxs:
            axs[*coords].remove()

        path = self.framesPaths.timeSeries_frame_path
        fig.savefig(path, dpi=DPI)
        LOG(f"[OK] {path}")


class FramesPaths:
    def __init__(self, context, userArgs):
        slice1_png_pattern = ""
        slice1Movie_path = context.runName
        if userArgs.zoom:
            slice1_png_pattern += f"zoom{userArgs.zoom}_"
            slice1Movie_path += f"zoom{userArgs.zoom}_"

        if userArgs.noBounds:
            slice1Movie_path += "unbounded"
            slice1Movie_path += "unbounded"

        slice1_png_pattern += "*.png"
        slice1Movie_path += ".mp4"
        self.slice1_png_pattern = context.slice1Folder / slice1_png_pattern
        self.slice1_video_path = context.videosFolder / slice1Movie_path

        self.spatiogram_frame_path = context.frameRootFolder / "spatiogram.png"
        self.timeSeries_frame_path = context.frameRootFolder / "timeSeries.png"


class Pipeline:
    def __init__(self, Context, UserArgs, zoom=0, end=1):
        """
        If there are n dumps, and end=0.5, only 0.5n of the dumps will be read.
        """
        self.userArgs = UserArgs
        self.context = Context
        self.end = end

        self.doParallel = True
        self.doSequential = False
        self.doMovie = True
        self.doStreamLines = True

        if self.userArgs.doOnlyFrames:  # safety guard
            self.doParallel = False
            self.doSequential = True
            self.doMovie = False

        self.processor = PhysicsProcessor(self.context, self.userArgs)

        self.fields1D = {}
        self.fields2D = {}
        self.partQuantities = {}
        if self.context.dimensions == 1:
            self.init_1D()
        elif self.context.dimensions == 2:
            self.init_2D()
        if self.context.outputTypes_info["particles"].status:
            self.init_particles()

        self.processor.set_fields(self.fields1D, self.fields2D)

        self._name_frames()

        self._apply_config()

    def _name_frames(self):
        context = self.context
        self.framesPaths = FramesPaths(context, self.userArgs)

        self.slice1_list = context.get_slice1_vtkFiles()
        self.vtkList = context.get_global_vtkFiles()
        self.partList = context.get_particles_vtkFiles()

    def _apply_config(self):
        if not self.userArgs.onlyMovie:
            if not self.userArgs.onlyAnalysis:
                all_fields = self.fields2D.keys()
                config = self.context.config
                print(config)
                all_bounds = {}

                if not self.userArgs.noBounds:
                    LOG("Computing bounds, please wait...")
                    fields_tobound = [
                        key
                        for key in all_fields
                        if key not in config or "range" not in config[key]
                    ]

                    if len(fields_tobound) > 0:
                        all_bounds = self.get_bounds(
                            self.slice1_list[min(len(self.slice1_list), 5) :],
                            fields_tobound,
                        )
                        print(fields_tobound)
                        [print(f"{key}: {all_bounds[key]}") for key in all_bounds]
                        LOG("Bounds computed")

                    else:
                        LOG("All fields are already bounded in config")

                    for qt in all_fields:
                        if qt in config and "range" in config[qt]:
                            self.fields2D[qt].set_bounds(config[qt]["range"])
                        elif qt in all_bounds and not self.userArgs.noBounds:
                            self.fields2D[qt].set_bounds(all_bounds[qt])
                        if qt in config:
                            if "cmap" in config[qt]:
                                self.fields2D[qt].set_cmap(config[qt]["cmap"])
                            if "norm" in config[qt]:
                                self.fields2D[qt].set_norm(config[qt]["norm"])
                else:
                    LOG("Bounds computation discarded.")

                LOG("Final Bounds:")
                for qt in self.fields2D:
                    LOG(qt, self.fields2D[qt].bounds)

    def init_2D(self):
        self.fields2D = {
            "VX1": Field2D("VX1", streamline="Vp", plot_coords=[1, 1]),
            "VX2": Field2D("VX2", streamline="Vp", plot_coords=[1, 2]),
            "VX3": Field2D("VX3", streamline="Vp", plot_coords=[1, 3]),
            "RHO": Field2D("RHO", streamline="Vp", plot_coords=[1, 4]),
            # "InvDt": Field2D(
            #     "InvDt", r"$\mathrm{d}t^{-1}$", streamline="Vp", plot_coords=[2, 0]
            # ),
        }
        vtkInfo = self.context.outputTypes_info["vtk"]

        if vtkInfo.dataHas["Pressure"]:
            self.fields2D["cs"] = Field2D("cs", streamline="Vp", plot_coords=[0, 4])
            self.fields2D["Mach_p"] = Field2D(
                "Mach_p", streamline="Vp", plot_coords=[0, 5]
            )
        if vtkInfo.dataHas["B"]:
            self.fields2D["BX1"] = Field2D("BX1", streamline="Bp", plot_coords=[0, 1])
            self.fields2D["BX2"] = Field2D("BX2", streamline="Bp", plot_coords=[0, 2])
            self.fields2D["BX3"] = Field2D("BX3", streamline="Bp", plot_coords=[0, 3])

            self.fields2D["eta"] = Field2D("eta", streamline="Vp", plot_coords=[0, 0])
            self.fields2D["Am"] = Field2D("Am", streamline="Vp", plot_coords=[1, 0])

        if vtkInfo.dataHas["Dust"]:
            self.fields2D["Dust0_VX1"] = Field2D(
                "Dust0_VX1", streamline="VpDust", plot_coords=[2, 1]
            )
            self.fields2D["Dust0_VX2"] = Field2D(
                "Dust0_VX2", streamline="VpDust", plot_coords=[2, 2]
            )
            self.fields2D["Dust0_VX3"] = Field2D(
                "Dust0_VX3", streamline="VpDust", plot_coords=[2, 3]
            )
            self.fields2D["Dust0_RHO"] = Field2D(
                "Dust0_RHO", streamline="VpDust", plot_coords=[2, 4]
            )
            # "St": Field2D("St", r"$\text{St}$", "Vp", plot_coords=[2, 0]),
        for key in self.fields2D:
            self.fields2D[key].name = register.alias(self.geometry, key)

        self.StreamLines = {
            "Bp": StreamLinesData("Bp", title=r"$\vec B_\text{p}$", color="#ccd5ae"),
            "Vp": StreamLinesData("Vp", title=r"$\vec V_\text{p}$", color="#adc178"),
            "VpDust": StreamLinesData(
                "VpDust", title=r"$\vec V_\text{p}^\text{dust}$", color="#dde5b6"
            ),
        }

        # At the moment, we only study the (x,z) plan.

    def init_1D(self):

        print(self.context.outputTypes_info["vtk"].dataHas["Dust"])
        if self.context.outputTypes_info["vtk"].dataHas["Dust"]:
            self.fields1D = {
                "Dust0_RHO": Field1D(
                    "Dust0_RHO",
                    r"$\rho^\text{dust}$",
                    plot_coords=[0, 0],
                    title=r"$\tau=1$",
                    id=1,
                ),
                "Dust1_RHO": Field1D(
                    "Dust1_RHO",
                    r"$\rho^\text{dust}$",
                    plot_coords=[0, 1],
                    title=r"$\tau=0.2$",
                    id=0.2,
                ),
                "Dust2_RHO": Field1D(
                    "Dust2_RHO",
                    r"$\rho^\text{dust}$",
                    plot_coords=[0, 2],
                    title=r"$\tau=0.04$",
                    id=0.04,
                ),
            }
        self.fields1D["RHO"] = Field1D(
            "RHO",
            r"$\rho$",
            plot_coords=[1, 2],
            title="Fluid",
        )

    def init_particles(self):
        self.partQuantities = {}
        vtkInfo = self.context.outputTypes_info["particles"]
        # assert vtkInfo.dataHas["Particles"]  # should be fine
        self.partQuantities["PART_X1"] = PartQuantity(
            "PART_X1", "PART_X1", plot_coords=[0, 0]
        )

    def get_bounds_indiv(self, args):
        """
        args (list[2]) must have two components:
            first:   vtk_path (str)
            second:   fields_indexes (dict) where fields_indexes[field] = index
        """
        vtk_path = args[0]
        fields_indexes = args[1]
        V = readVTK(vtk_path)
        self.processVTK(V)
        bounds = np.empty((len(fields_indexes), 2))
        for field in fields_indexes.keys():
            data = V.data[field]
            index = fields_indexes[field]
            bounds[index, 0] = np.nanmin(data)
            bounds[index, 1] = np.nanmax(data)
        return bounds

    def get_bounds(self, vtkList, fields):
        """
        Get the bounds (min, max) of all given fields. I recommend not passing the entire vtkList but rather vtkList[1:] to discard the first output(s ?).

        vtkList    List of dump file paths
        field       Field

        returns
        dict where dict[field] = (min, max)
        """
        mapfields_indexes = {}
        for i, field in enumerate(fields):
            mapfields_indexes[field] = i

        with Pool(CPU) as pool:
            all_bounds = pool.map(
                self.get_bounds_indiv,
                [[vtk, mapfields_indexes] for vtk in vtkList],
            )
        all_bounds = np.array(all_bounds)
        bounds = {}
        if len(all_bounds) == 0:
            return bounds
        for field in fields:
            i = mapfields_indexes[field]
            bounds[field] = (
                np.nanmin(all_bounds[:, i, 0]),
                np.nanmax(all_bounds[:, i, 1]),
            )
        return bounds

    # def plot_slice(self):

    #     if self.userArgs.doOnlyFrames:
    #         for vtkPath in [self.slice1_list[i] for i in self.userArgs.doOnlyFrames]:
    #             self.slice_to_png(vtkPath)
    #     else:
    #         if self.doParallel:
    #             with Pool(CPU) as pool:
    #                 pool.map(self.slice_to_png, self.slice1_list)
    #         else:
    #             fields1D_result = []
    #             for vtkPath in self.slice1_list:
    #                 result = self.slice_to_png(vtkPath)
    #                 fields1D_result.append(result)

    #     if self.doMovie and not self.userArgs.onlyAnalysis:
    #         tools.movie(
    #             self.slice1_png_pattern,
    #             self.slice1Movie_path,
    #         )

    def run(self):
        # if self.context.dimensions == 1:
        # Field1D
        # if self.doParallel:
        #     with Pool(CPU) as pool:
        #         fields1D_result = pool.starmap(
        #             self.processor.get_quantities,
        #             zip(self.vtkList, repeat(self.fields1D)),
        #         )
        #     nb_vtkTimes = len(fields1D_result)
        #     vtkTimes = [fields1D_result[i][0] for i in range(nb_vtkTimes)]
        #     for field in self.fields1D.values():
        #         values = np.array(
        #             [fields1D_result[i][field.index] for i in range(nb_vtkTimes)]
        #         )
        #         # values = np.where(values < 1, values, np.nan)
        #         # values = np.where(values > 0, values, np.nan)
        #         field.set_data(points=self.processor.X1Line, values=values)
        #     self.context.outputTypes_info["vtk"].set_Times(vtkTimes)
        # vtkInfo = self.context.outputTypes_info["vtk"]
        # if vtkInfo.dataHas["Dust"]:
        #     integrateTimes = np.linspace(vtkInfo.tStart, vtkInfo.tEnd, 1000)
        #     for dustField in ["Dust0_RHO", "Dust1_RHO", "Dust2_RHO"]:
        #         field = self.fields1D[dustField]
        #         Stokes0 = field.id
        #         fluid = dust.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
        #         r0 = 2
        #         predictedTrajectory = tools.integrate(fluid.vrDrift, r0, integrateTimes)
        #         field.set_ref_data(integrateTimes, predictedTrajectory)

        partInfo = self.context.outputTypes_info["particles"]
        if partInfo.status:
            if self.doParallel:
                with Pool(CPU) as pool:
                    particles_result = pool.starmap(
                        self.processor.get_quantities,
                        zip(self.partList, repeat(self.partQuantities)),
                    )
            nb_vtkTimes = len(particles_result)
            Times = [particles_result[i][0] for i in range(nb_vtkTimes)]
            for qty in self.partQuantities.values():
                values = np.array(
                    [particles_result[i][qty.index] for i in range(nb_vtkTimes)]
                )
                qty.set_data(points=Times, values=values)
            self.context.outputTypes_info["particles"].set_Times(Times)

            for partField in ["PART_X1"]:
                field = self.partQuantities[partField]
                Stokes0 = 1
                fluid = dust.Fluid(0.05, -0.5, 0.125, -0.5, Stokes0=Stokes0)
                r0 = 2
                integrateTimes = np.linspace(partInfo.tStart, partInfo.tEnd, 1000)
                predictedTrajectory = tools.integrate(fluid.vrDrift, r0, integrateTimes)
                field.set_ref_data(integrateTimes, predictedTrajectory)

        renderer = SliceRenderer(
            self.context,
            self.processor,
            self.fields1D,
            self.fields2D,
            self.partQuantities,  # TODO add quantities eventually
            self.userArgs,
            self.framesPaths,
        )

        if self.context.outputTypes_info["particles"].status:
            renderer.render_TimeSeries()
        # renderer.render_Spatiogram()  # TODO Give the choice between spatiogram or 1Dmovie
        # with Pool(CPU) as pool:
        #     pool.map(renderer.render1D, self.vtkList)

        # print(self.framesPaths.slice1_video_path)
        # tools.movie(
        #     self.framesPaths.slice1_png_pattern,
        #     self.framesPaths.slice1_video_path,
        #     fps=2,
        # )
        # TODO
        # if self.dimensions == 2:


def do_task(task, args):
    # projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/DriftSettling"
    projectPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift"
    configPath = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config.json"

    runContext = RunContext(projectPath, task, args, configPath=configPath)
    # run.plot_slice()
    pipeline = Pipeline(runContext, args)
    pipeline.run()
    # run.plot_1Devolution_particles()
    # run.plot_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--frame",
        nargs="*",
        default=None,
        help="integer: will only render this frame",
        type=int,
        dest="doOnlyFrames",
    )

    parser.add_argument(
        "-z",
        "--zoom",
        nargs="?",
        const=0,
        default=0,
        help="float: will only render r < zoom",
        type=float,
    )

    parser.add_argument(
        "--no-bounds",
        action="store_true",
        dest="noBounds",
        help="will ignore the config file and let free bounds on colorbars",
    )

    parser.add_argument(
        "-om", action="store_true", help="only movie?", dest="onlyMovie"
    )

    parser.add_argument(
        "-oa", action="store_true", help="only analysis?", dest="onlyAnalysis"
    )

    args = parser.parse_args()
    if args.doOnlyFrames is None:
        args.doOnlyFrames = False
    elif len(args.doOnlyFrames) == 0:
        args.doOnlyFrames = [0]

    print(args)

    # tasks = ["DS_test"]
    tasks = ["DriftL_Tau"]
    for task in tasks:
        do_task(task, args)
