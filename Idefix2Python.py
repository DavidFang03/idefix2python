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


plt.style.use("dark_background")
# plt.rcParams.update(
#     {
#         # "text.usetex": True,
#     }
# )
plt.rcParams["hatch.color"] = "gray"
plt.rcParams["hatch.linewidth"] = 0.5

DEFAULT_CMAP = "berlin"

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


class Data_info:
    """
    Different types of output: global (vtk), slice (vtk), timevol (dat)
    """

    def __init__(self, test_file_pattern):
        self.test_files = glob.glob(test_file_pattern)
        self.geometry = None
        if len(self.test_files) > 0:
            self.status = True
            self.test_file = self.test_files[0]
            self.ext = self.test_file[-4:]

            self.set_data_test()

        else:
            self.status = False

    def set_data_test(self):
        if "vtk" in self.ext:
            vtk = readVTK(self.test_file)
            self.data_test = vtk.data
            self.geometry = vtk.geometry
            self.vtk = vtk
        elif "dat" in self.ext:
            self.data_test = tools.dat_to_dict(self.test_file)


class Scalar:
    def __init__(self, key, symbol, **kwargs):
        scale = kwargs.get("scale", "linear")
        title = kwargs.get("title", symbol)
        self.key = key
        self.symbol = symbol
        self.title = title
        self.scale = scale

    def set_data(self, data):
        self.data = data


class Field1DwithTime:
    def __init__(self, key, symbol, index, **kwargs):
        scale = kwargs.get("scale", "linear")
        title = kwargs.get("title", symbol)
        self.key = key
        self.symbol = symbol
        self.title = title
        self.scale = scale
        self.index = index

    def set_data(self, points, values):
        self.points = points
        self.values = values


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


class Quantity2D:
    def __init__(self, key, streamline=None, plot_coords=None):
        self.key = key
        self.name = key
        self.streamline = streamline

        self.bounds = [None, None]
        self.cmap = DEFAULT_CMAP
        self.norm = None

        if plot_coords is not None:
            self.set_plot_coords(*plot_coords)

    def set_name(self, name):
        self.name = name

    def set_data(self, data):
        self.data = data

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_cmap(self, cmap):
        self.cmap = cmap

    def set_norm(self, norm):
        self.norm = norm

    def set_plot_coords(self, i, j):
        self.coords = [i, j]


class RUN:
    def __init__(self, PathToProject, RunName, args, configs, zoom=0, end=1):
        """
        If there are n dumps, and end=0.5, only 0.5n of the dumps will be read.
        """
        self.zoom = args.zoom
        self.unbounded = args.freeBounds
        self.onlyAnalysis = args.oa
        self.onlyMovie = args.om
        self.doParallel = True
        self.doMovie = True

        self.doOnlyFrames = args.frame
        if self.doOnlyFrames:  # safety guard
            self.doParallel = False
            self.doSequential = True
            self.doMovie = False
        self.doStreamLines = True

        tools.RequirePath(PathToProject, dir_or_file="dir")
        self.DataPath = f"{PathToProject}/outputs/{RunName}"
        self.iniPath = f"{PathToProject}/inputs/{RunName}.ini"
        self.name = RunName
        self.formatInputs_text = tools.formatInputs(self.iniPath)
        if self.name in configs:
            self.config = configs[self.name]
        else:
            self.config = configs["default"]

        # Data location
        self.analysis_path = f"{self.DataPath}/timevol.dat"
        self.vtk_pattern = f"{self.DataPath}/vtks/data*.vtk"
        self.slice1_pattern = self.vtk_pattern

        # Renders location
        self.globalFrames_folder = f"{PathToProject}/frames/{RunName}/global"
        self.slice1Frames_folder = self.globalFrames_folder  # dirty trick
        self.singleFrame_folder = f"{PathToProject}/frames/{RunName}"
        self.videos_folder = f"{PathToProject}/videos"
        self.analysisFrame_path = f"{self.singleFrame_folder}/analysis.png"
        self.slice1Movie_path = f"{self.videos_folder}/{self.name}_slice1"

        self.slice1_png_pattern = f"{self.slice1Frames_folder}/"
        if self.zoom:
            self.slice1_png_pattern += f"zoom{self.zoom}_"
            self.slice1Movie_path += f"_zoom{self.zoom}"
        else:
            self.slice1_png_pattern += "data"
        self.slice1_png_pattern += "*.png"

        if self.unbounded:
            self.slice1Movie_path += "_unbounded"
        self.slice1Movie_path += ".mp4"

        self.slice1_list = sorted(glob.glob(self.slice1_pattern))
        self.slice1_list = self.slice1_list[: int(end * len(self.slice1_list))]

        self.vtk_list = sorted(glob.glob(self.vtk_pattern))
        self.vtk_list = self.vtk_list[: int(end * len(self.vtk_list))]

        self.data_info = {}
        self.data_types = ["analysis", "slice1", "vtk"]
        self.data_info["analysis"] = Data_info(self.analysis_path)
        self.data_info["slice1"] = Data_info(self.slice1_pattern)
        self.data_info["vtk"] = Data_info(self.vtk_pattern)

        self.quantities2D = {
            "VX1": Quantity2D("VX1", "Vp", plot_coords=[1, 1]),
            "VX2": Quantity2D("VX2", "Vp", plot_coords=[1, 2]),
            "VX3": Quantity2D("VX3", "Vp", plot_coords=[1, 3]),
            "RHO": Quantity2D("RHO", "Vp", plot_coords=[1, 4]),
            # "InvDt": Quantity2D(
            #     "InvDt", r"$\mathrm{d}t^{-1}$", "Vp", plot_coords=[2, 0]
            # ),
        }

        self.dataHas = {
            "Pressure": False,
            "B": False,
            "Dust": False,
            "Particles": False,
        }

        self.geometry = None
        self.check_available_quantities()
        if self.dataHas["Pressure"]:
            self.quantities2D["cs"] = Quantity2D("cs", "Vp", plot_coords=[0, 4])
            self.quantities2D["Mach_p"] = Quantity2D("Mach_p", "Vp", plot_coords=[0, 5])
        if self.dataHas["B"]:
            self.quantities2D["BX1"] = Quantity2D("BX1", "Bp", plot_coords=[0, 1])
            self.quantities2D["BX2"] = Quantity2D("BX2", "Bp", plot_coords=[0, 2])
            self.quantities2D["BX3"] = Quantity2D("BX3", "Bp", plot_coords=[0, 3])

            self.quantities2D["eta"] = Quantity2D("eta", "Vp", plot_coords=[0, 0])
            self.quantities2D["Am"] = Quantity2D("Am", "Vp", plot_coords=[1, 0])

        if self.dataHas["Dust"]:
            self.quantities2D["Dust0_VX1"] = Quantity2D(
                "Dust0_VX1", "VpDust", plot_coords=[2, 1]
            )
            self.quantities2D["Dust0_VX2"] = Quantity2D(
                "Dust0_VX2", "VpDust", plot_coords=[2, 2]
            )
            self.quantities2D["Dust0_VX3"] = Quantity2D(
                "Dust0_VX3", "VpDust", plot_coords=[2, 3]
            )
            self.quantities2D["Dust0_RHO"] = Quantity2D(
                "Dust0_RHO", "VpDust", plot_coords=[2, 4]
            )
            # "St": Quantity2D("St", r"$\text{St}$", "Vp", plot_coords=[2, 0]),
        for key in self.quantities2D:
            self.quantities2D[key].set_name(register.alias(self.geometry, key))

        self.StreamLines = {
            "Bp": StreamLinesData("Bp", title=r"$\vec B_\text{p}$", color="#ccd5ae"),
            "Vp": StreamLinesData("Vp", title=r"$\vec V_\text{p}$", color="#adc178"),
            "VpDust": StreamLinesData(
                "VpDust", title=r"$\vec V_\text{p}^\text{dust}$", color="#dde5b6"
            ),
        }

        # >>>>>>>>> Post Timeseries <<<<<<<<<<<< #
        self.PostTimeSeries = []  # shape (N_timesteps,N_quantities)
        self.Scalars = {
            "divB": Scalar("divB", r"$\mathrm{div} B$", scale="linear"),
            "mass": Scalar("mass", r"$M$", title="Mass (normalized)", scale="log"),
        }
        self.Fields1DwithTime = {}
        # if self.dataHas["Dust"]:
        #     self.Fields1DwithTime = {
        #         "Vr_avg": Field1DwithTime("Vr_avg", r"$\overline{v}_R$", index=1),
        #     }
        # self.Scalars["Vr_avg"].set_index(1)
        # we must create
        # ({project}/frames/{RunName})
        # {project}/frames/{RunName}/global
        # {project}/frames/{RunName}/slice1
        # {project}/videos/{RunName}
        folders = [
            self.globalFrames_folder,
            self.slice1Frames_folder,
            self.videos_folder,
        ]
        for path in folders:
            try:
                print(f"mkdir {path}")
                os.makedirs(path)
            except OSError as _:
                pass
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

        # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
        if self.data_info["slice1"].status:
            vtk = self.data_info["slice1"].vtk

            if self.geometry == "cartesian":
                self.X1Line, self.X2Line = vtk.x, vtk.z
            elif self.geometry == "cylindric":
                self.X1Line, self.X2Line = vtk.r, vtk.z
            elif self.geometry == "polar":
                raise NotImplementedError("POLAR geometry not implemented yet")
            elif self.geometry == "spherical":
                self.X1Line, self.X2Line = vtk.r, vtk.theta

            # At the moment, we only study the (x,z) plan.

            self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)

            self.X, self.Z = tools.convertGrid_toXZ(self.X1, self.X2, self.geometry)

            if not self.zoom:
                self.mask = np.full(self.X.shape, True, dtype=bool)
            else:
                self.mask = (
                    (self.X < self.zoom) & (np.abs(self.Z) < self.zoom)
                    # & (np.abs(np.pi / 2 - self.Theta) > np.pi / 12)
                )
            self.xmin = 0
            self.xmax = np.max(np.where(self.mask, self.X, 0))
            self.ymax = np.max(np.where(self.mask, self.Z, 0))
            self.ymin = np.min(np.where(self.mask, self.Z, 0))

    def check_available_quantities(self):
        """
        Show quantities in every kind of data and detect is there are Pressure, B, Dust or Particles fields. Also detect the geometry.
        """
        print(f"------ Available quantities in {self.DataPath} ------")
        for data_type in self.data_types:
            if self.data_info[data_type].status:
                print(f"{data_type}:", self.data_info[data_type].data_test.keys())
                for qt in self.data_info[data_type].data_test.keys():
                    print(
                        f"{qt:>10} {np.shape(self.data_info[data_type].data_test[qt].data)}"
                    )
                    if qt == "PRS":
                        self.dataHas["Pressure"] = True
                    elif qt == "BX1":
                        self.dataHas["B"] = True
                    elif qt == "Dust0_RHO":
                        self.dataHas["Dust"] = True
                    elif qt == "PX1":
                        self.dataHas["Particles"] = True

            geometry = self.data_info[data_type].geometry
            if geometry is not None:
                self.geometry = geometry

    def plot_analysis(self):
        self.analysis = self.data_info["analysis"].data_test
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(self.DataPath)
        fig.subplots_adjust(left=0.1, right=1 - 0.05, top=0.75, hspace=0.3, wspace=0.2)
        # a = self.analysis
        i = 0
        # t = a["t"]

        # for timeseries in self.Scalars:
        #     i += 1
        #     ax = axs[i % 3, i // 3]
        #     qty1D_info = self.Scalars[timeseries]
        # ax.plot(t, a[timeseries], label=qty1D_info.title)
        #     ax.set_xlabel(r"$t$")
        #     ax.set_ylabel(qty1D_info.symbol)
        #     ax.set_yscale(qty1D_info.scale)
        #     ax.set_title(qty1D_info.title)
        #     ax.legend()
        #     ax.grid()

        for key, field in self.Fields1DwithTime.items():
            i += 1
            ax = axs[i % 3, i // 3]
            T, Points = np.meshgrid(np.asarray(self.vtkTimes), np.asarray(field.points))
            cmesh = ax.pcolormesh(
                T, Points, np.transpose(field.values), shading="nearest"
            )
            cbar = fig.colorbar(cmesh, ax=ax)
            cbar_title = f"{field.title}"
            cbar.ax.set_title(cbar_title)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$R$")
            ax.set_title("Averaged radial velocity")
            ax.grid()

        tools.annotateInputs(axs, self.formatInputs_text)

        image_path = self.analysisFrame_path
        fig.savefig(image_path)
        print(f"[OK] {image_path}")

    def processVTK(self, V):
        """
        Transposes the vtk datas and add some stuff:
            - Soundspeed
            - Mach number
        """
        for qt in V.data:
            V.data[qt] = np.transpose(V.data[qt][:, :, 0])
            V.data[qt] = np.where(self.mask, V.data[qt], np.nan)

        if self.dataHas["Pressure"]:
            V.data["cs"] = tools.divide_discardingNullDenominator(
                np.sqrt(V.data["PRS"]), np.sqrt(V.data["RHO"])
            )
            V.data["Mach_p"] = tools.divide_discardingNullDenominator(
                np.sqrt(V.data["VX1"] ** 2 + V.data["VX2"] ** 2),
                V.data["cs"],
            )

        # For streamlines
        V.data["vx"], V.data["vz"] = tools.convertVector_toXZ(
            V.data["VX1"],
            V.data["VX2"],
            self.X1,
            self.X2,
            self.geometry,
        )
        if self.dataHas["B"]:
            V.data["Bx"], V.data["Bz"] = tools.convertVector_toXZ(
                V.data["BX1"],
                V.data["BX2"],
                self.X1,
                self.X2,
                self.geometry,
            )
        if self.dataHas["Dust"]:
            V.data["vxDust"], V.data["vzDust"] = tools.convertVector_toXZ(
                V.data["Dust0_VX1"],
                V.data["Dust0_VX1"],
                self.X1,
                self.X2,
                self.geometry,
            )

        ## Post Time Series
        # Vr_avg
        # V.data["Vr_avg"] = np.average(V.data["VX1"], axis=0)

        # Reynolds number
        # X = self.X
        # epsilon = 0.1
        # Omega = np.pow(X, -1.5)
        # V.data["Rm"] = divide_discardingNullDenominator(
        #     Omega * np.pow(epsilon * X, 2), V.data["eta"]
        # )
        # V.data["rho"] = np.log10(V.data["RHO"])

        # V.data["Mach_p"] = applyOperation_discardingNone(np.log10, V.data["Mach_p"])
        # V.data["Mach_p"] = np.log10(V.data["Mach_p"])

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

    def get_bounds(self, vtk_list, fields):
        """
        Get the bounds (min, max) of all given fields. I recommend not passing the entire vtk_list but rather vtk_list[1:] to discard the first output(s ?).

        vtk_list    List of dump file paths
        field       Field

        returns
        dict where dict[field] = (min, max)
        """
        mapfields_indexes = {}
        for i, field in enumerate(fields):
            mapfields_indexes[field] = i

        with Pool(16) as pool:
            all_bounds = pool.map(
                self.get_bounds_indiv,
                [[vtk, mapfields_indexes] for vtk in vtk_list],
            )
        # print(np.shape(all_bounds))
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

    def slice_to_png(self, slice1_path):
        V = readVTK(slice1_path)
        X = self.X
        Z = self.Z
        self.processVTK(V)
        time = V.t[0]

        if not self.onlyAnalysis:
            rows = 3
            columns = (
                max([qtyInfo.coords[1] for qtyInfo in self.quantities2D.values()]) + 1
            )
            cbars = {}
            fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 16))
            if self.zoom:
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

            fig.suptitle(f"{self.name}\n{slice1_path}\n$t={time:.1e}$")

            tools.annotateInputs(axs, self.formatInputs_text)

            if self.doStreamLines:
                self.StreamLines["Vp"].set_data(
                    *get_streamplot_data(
                        self.X1Line,
                        self.X2Line,
                        V.data["vx"],
                        V.data["vz"],
                        self.geometry,
                        self.xmax,
                    )
                )
                if self.dataHas["B"]:
                    self.StreamLines["Bp"].set_data(
                        *get_streamplot_data(
                            self.X1Line,
                            self.X2Line,
                            V.data["Bx"],
                            V.data["Bz"],
                            self.geometry,
                            self.xmax,
                        )
                    )
                if self.dataHas["Dust"]:
                    self.StreamLines["VpDust"].set_data(
                        *get_streamplot_data(
                            self.X1Line,
                            self.X2Line,
                            V.data["vxDust"],
                            V.data["vzDust"],
                            self.xmax,
                        )
                    )

            unusedAxs = [[i, j] for i in range(rows) for j in range(columns)]
            for qty, qtyInfo in self.quantities2D.items():
                data = V.data[qty]
                ax = axs[*qtyInfo.coords]
                unusedAxs.remove(qtyInfo.coords)
                streamline = qtyInfo.streamline
                if None in qtyInfo.bounds or self.unbounded:
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
                elif qtyInfo.norm == "TwoSlopeNorm" and not self.unbounded:
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

                if self.zoom:
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
                ax.set_xlim(self.xmin, self.xmax)
                ax.set_ylim(self.ymin, self.ymax)

                if qtyInfo.coords[1] == 0:
                    ax.set_ylabel(r"$z$")
                title = qtyInfo.name
                color = None

                if self.doStreamLines and streamline:
                    streamlineData = self.StreamLines[streamline]
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

            if self.dataHas["Pressure"]:
                Mach_pInfo = self.quantities2D["Mach_p"]
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

            slice1_name = os.path.basename(slice1_path)
            slice1_png_path = self.slice1_png_pattern.replace(
                "*", f"{slice1_name[:-4]}"
            )
            fig.savefig(slice1_png_path, dpi=300)
            plt.close(fig)
            print(f"[OK] {slice1_png_path}")

        # Post Time Series
        PostTimeSeries = [None for _ in range(1 + len(self.Fields1DwithTime))]
        PostTimeSeries[0] = V.t[0]
        for key, field in self.Fields1DwithTime.items():
            PostTimeSeries[field.index] = V.data[key]
        return PostTimeSeries

    def plot_slice(self):
        if not self.onlyMovie:
            if not self.onlyAnalysis:
                all_quantities = self.quantities2D.keys()
                config = self.config
                print(config)
                all_bounds = {}

                if not self.unbounded:
                    print("Computing bounds, please wait...")
                    quantities_tobound = [
                        key
                        for key in all_quantities
                        if key not in config or "range" not in config[key]
                    ]

                    if len(quantities_tobound) > 0:
                        all_bounds = self.get_bounds(
                            self.slice1_list[min(len(self.slice1_list), 5) :],
                            quantities_tobound,
                        )
                        print(quantities_tobound)
                        [print(f"{key}: {all_bounds[key]}") for key in all_bounds]
                        print("Bounds computed")

                    else:
                        print("All quantities are already bounded in config")

                    for qt in all_quantities:
                        if qt in config and "range" in config[qt]:
                            self.quantities2D[qt].set_bounds(config[qt]["range"])
                        elif qt in all_bounds and not self.unbounded:
                            self.quantities2D[qt].set_bounds(all_bounds[qt])
                        if qt in config:
                            if "cmap" in config[qt]:
                                self.quantities2D[qt].set_cmap(config[qt]["cmap"])
                            if "norm" in config[qt]:
                                self.quantities2D[qt].set_norm(config[qt]["norm"])
                else:
                    print("Bounds computation discarded.")

                print("Final Bounds:")
                for qt in self.quantities2D:
                    print(qt, self.quantities2D[qt].bounds)

            if self.doOnlyFrames:
                for slice1_path in [self.slice1_list[i] for i in self.doOnlyFrames]:
                    self.slice_to_png(slice1_path)
            else:
                if self.doParallel:
                    with Pool(16) as pool:
                        Fields1DwithTime_result = pool.map(
                            self.slice_to_png, self.slice1_list
                        )
                else:
                    Fields1DwithTime_result = []
                    for slice1_path in self.slice1_list:
                        result = self.slice_to_png(slice1_path)
                        Fields1DwithTime_result.append(result)

                nb_vtkTimes = len(Fields1DwithTime_result)
                self.vtkTimes = [
                    Fields1DwithTime_result[i][0] for i in range(nb_vtkTimes)
                ]
                for field in self.Fields1DwithTime.values():
                    values = [
                        Fields1DwithTime_result[i][field.index]
                        for i in range(nb_vtkTimes)
                    ]
                    field.set_data(points=self.X1Line, values=values)

        if self.doMovie and not self.onlyAnalysis:
            tools.movie(
                self.slice1_png_pattern,
                self.slice1Movie_path,
            )


def do_task(task, args):
    PathToProject = "/home/dp316/dp316/dc-fang1/IdefixRuns/DriftSettling"
    config_path = "/home/dp316/dp316/dc-fang1/IdefixRuns/Idefix2Python/config.json"

    configs = tools.process_configs(config_path)

    run = RUN(PathToProject, task, args, configs, end=1)
    run.plot_slice()
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
        dest="freeBounds",
        help="will ignore the config file and let free bounds on colorbars",
    )

    parser.add_argument(
        "-om",
        action="store_true",
        help="only movie?",
    )

    parser.add_argument(
        "-oa",
        action="store_true",
        help="only analysis?",
    )

    args = parser.parse_args()
    if args.frame is None:
        args.frame = False
    elif len(args.frame) == 0:
        args.frame = [0]

    print(args)

    tasks = ["DS_test_reflective_alpha1e-2"]
    for task in tasks:
        count = 0
        do_task(task, args)
