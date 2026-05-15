import os
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat
import argparse

from .tools import LOG
from . import tools
from .vtk_io import readVTK
from .renderer import SliceRenderer
from .processor import PhysicsProcessor, PartsInfo
from .quantities import PartQuantity, SpaceTimeHeatmap, MapMovie2D, LineMovie1D


class OutputTypeInfo:
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
            self.vtk = vtk
            if "part" not in str(self.test_file):
                self.dimensions = vtk.dimensions

        elif "dat" in self.ext:
            self.testData = tools.dat_to_dict(self.test_file)

    def get_availableKeys(self):
        if not self.status:
            return f"No {self.name} present"
        LOG(f"------ Available fields in {self.test_file} ------")

        for qt in self.testData.keys():
            LOG(f"{qt:>10} {np.shape(self.testData[qt].data)}")
            if qt == "PRS":
                self.dataHas["Pressure"] = True
            elif qt.startswith("BX"):
                self.dataHas["B"] = True
            elif qt.startswith("Dust"):
                self.dataHas["Dust"] = True
            elif qt.startswith("PART"):
                self.dataHas["Particles"] = True
        if "vtk" in self.ext:
            dataStart = self.files[0]
            dataEnd = self.files[-1]  # TODO end option
            self.tStart = readVTK(dataStart).t[0]
            self.tEnd = readVTK(dataEnd).t[0]

        elif "dat" in self.ext:
            raise NotImplementedError()


class RunContext:
    """
    The first thing to initiate.

    Handles data location and directory creation. Detects simulation geometry,
    dimensions, and available fields.

    Args:
        runName (str): The unique name of the run.
        projectPath (str | Path, optional): The root directory of the project.
            Defaults to the current directory (".").
        **kwargs: Additional optional parameters:

            * configPath (str | Path): Path to a specific configuration file.
            * partFolder (str): Folder path containing the particles data.
            * frameFolder (str): Folder name where the rendered frames will be stored.
            * active_directions (list): List of active coordinate directions.
            * debug (bool): debug mode will show the .ini file.
                Defaults to False.
            * iniPath (Path): Custom path to the .ini input file. Defaults to
              `projectPath/inputs/{runName}.ini`.

    Note:
        The expected location for the .vtk files is `projectPath/outputs/runName/vtks`.
        By default, the rendered frame will be located in `projectPath/frames/runName`.
    """

    def __init__(self, runName, projectPath=".", **kwargs):
        self.runName = runName
        self.projectPath = Path(projectPath)
        self.projectPath.resolve(strict=True)

        self.debug = kwargs.get("debug", False)

        self.args = kwargs.get("args", _get_args())

        self.config = {}
        configPath = kwargs.get("configPath", None)
        self.configPath = configPath
        if configPath is not None:
            self.config = tools.process_configs(configPath)

        self.dataPath = self.projectPath / "outputs" / runName
        self.iniPath = Path(
            kwargs.get("iniPath", self.projectPath / "inputs" / f"{runName}.ini")
        )
        self.format_inputs_text = ""
        if self.debug:
            if self.iniPath.is_file():
                self.format_inputs_text = tools.formatInputs(self.iniPath)
            else:
                raise FileNotFoundError(
                    f"debug requested but {self.iniPath} doesn't exist"
                )

        self.partFolder = kwargs.get("partFolder", None)

        self.frameFolderName = kwargs.get("frameFolder", runName)
        self.active_directions = kwargs.get("active_directions", [])
        # for part*.vtk, the readVTK routine can't deduce the number of dimensions.
        # The context will try to find the dimensions in data*.vtk
        # If there is no data*.vtk, the user has to pass the dimensions.

        self._setup_directories()
        self._check_data()

    def _setup_directories(self):
        self.frameRootFolder = self.projectPath / "frames" / self.frameFolderName
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
        # self.outputTypes_info["analysis"] = OutputTypeInfo(self.analysis_path, "analysis")
        self.outputTypes_info["vtk"] = OutputTypeInfo("vtk", self.get_global_vtkFiles())
        self.outputTypes_info["slice1"] = OutputTypeInfo(
            "slice1", self.get_slice1_vtkFiles()
        )
        self.outputTypes_info["particles"] = OutputTypeInfo(
            "particles", self.get_particles_vtkFiles()
        )
        self.outputTypes_info["particles"].dimensions = self.outputTypes_info[
            "vtk"
        ].dimensions
        # There's no way to deduce the number of dimensions from the part*.vtk files but it has to be the same as in the global vtk

        if (
            self.partFolder is not None
            and not self.outputTypes_info["particles"].status
        ):
            raise FileNotFoundError(
                f"the folder {self.partFolder} doesn't seem to contain any part*vtk"
            )

        ## Everything is deduced from the global vtk
        vtkInfo = self.outputTypes_info["vtk"]
        partInfo = self.outputTypes_info["particles"]
        if vtkInfo.status:
            geometry = vtkInfo.geometry
        elif partInfo.status:
            geometry = partInfo.geometry
        else:
            raise Exception("No vtk files were found?")

        if len(self.active_directions) == 0:
            if not vtkInfo.status:
                raise Exception(
                    "No data*.vtk detected. Please provide active_directions."
                )
            vtk = vtkInfo.vtk
            for direction, ncell in enumerate([vtk.nx, vtk.ny, vtk.nz]):
                if ncell > 1:
                    self.active_directions.append(direction)

        dimensions = len(self.active_directions)
        self.geometry = geometry
        self.dimensions = dimensions

        self.active_directions_labels = [
            tools.get_Position_name(self.geometry, dir)
            for dir in self.active_directions
        ]
        LOG("Dimensions detected: ", self.dimensions)
        LOG("Active axes", self.active_directions_labels)

        if self.outputTypes_info["particles"].status:
            self.all_particles_uids = self.outputTypes_info["particles"].testData["uid"]
            self.particles_nb = len(self.all_particles_uids)
            LOG(f"Particles detected: {self.particles_nb}")
        else:
            self.particles_nb = 0
            self.all_particles_uids = []

    def _get_lastfile_to_read(self, filelist):
        """
        expects a sorted list
        """
        until = self.args.until
        if len(filelist) == 0:
            lastframe = -1
        elif 0 <= until <= 1:
            lastframe = int(len(filelist) * until)
        elif isinstance(until, int):
            lastframe = until
        elif isinstance(until, float):
            if str(filelist[-1]).endswith(".vtk"):
                tend = readVTK(filelist[-1]).t[0]
                tstart = readVTK(filelist[0]).t[0]
                if until < tstart:
                    raise Exception(
                        f"Value of until ({until}) is inferior than the first file time"
                    )
                lastframe = int((until - tstart) / (tend - tstart) * len(filelist))
                if lastframe + 1 < len(filelist):
                    lastframe += 1
        return lastframe

    def get_global_vtkFiles(self):
        pattern = "vtks/data*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.args.every]

    def get_slice1_vtkFiles(self):
        pattern = "vtks/slice1*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.args.every]

    def get_particles_vtkFiles(self):
        if self.partFolder is not None:
            filelist = sorted(Path(self.partFolder).glob("part*.vtk"))
        else:
            pattern = "vtks/part*.vtk"
            filelist = sorted(self.dataPath.glob(pattern))

        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.args.every]


class Pipeline:
    def __init__(
        self,
        Context,
        figs,
        zoom=0,
        streamLines=None,
    ):
        """
        Coordinates the detection, processing, and rendering of the simulation data.
        :param Context: The RunContext object containing simulation metadata.
        :type Context: RunContext
        :param spaceTimeHeatmaps: Objects defining (x, t) heatmap plots.
        :type spaceTimeHeatmaps: list[SpaceTimeHeatmap], optional
        :param movies1D: Objects defining 1D line plot animations.
        :type movies1D: list[LineMovie1D], optional
        :param movies2D: Objects defining 2D heatmaps animations.
        :type movies2D: list[MapMovie2D], optional
        :param partQuantities: Objects defining particles quantities.
        :type partQuantities: list[PartQuantity], optional
        :param zoom: Zoom level for the rendering view (for 2D only currently).
        :type zoom: float, optional
        :param streamLines: Configuration for streamlines overlays.
        :type streamLines: StreamlineConfig, optional
        """
        self.context = Context
        self.userArgs = self.context.args

        self.doMovie = True

        self.streamLines = streamLines

        self.processor = PhysicsProcessor(self.context, self.userArgs, self.streamLines)

        self.figs = figs
        self.spaceTimeHeatmaps = []
        self.partQuantities = []
        self.movies1D = []
        self.movies2D = []

        qty_tocompute = []
        self.particles_requested = False
        for fig in figs:
            for qtyInfo in fig.quantities:
                if isinstance(qtyInfo, PartQuantity):
                    self.partQuantities.append(qtyInfo)
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    self.spaceTimeHeatmaps.append(qtyInfo)
                elif isinstance(qtyInfo, LineMovie1D):
                    self.movies1D.append(qtyInfo)
                elif isinstance(qtyInfo, MapMovie2D):
                    self.movies2D.append(qtyInfo)

                if qtyInfo.compute is not None:
                    qty_tocompute.append(qtyInfo)

                if qtyInfo.uids is not None:
                    self.particles_requested = True
                    if qtyInfo.uids == "all":
                        qtyInfo.uids = self.context.all_particles_uids

        self.processor.partsInfo = PartsInfo(
            self.context.active_directions
        )  # pipeline bro helping clueless processor

        if self.particles_requested:
            self.partQuantities += self.processor.partsInfo.partsqty_togather

        self.processor.set_qty_tocompute(qty_tocompute)
        self.processor.set_partQuantities(self.partQuantities)

        self.renderer = SliceRenderer(
            self.context,
            self.processor,
            self.figs,
            self.userArgs,
        )

        self._name_frames()
        self._apply_config()

    def _check_everything_alright(self):
        # Check whether the particles requested exist
        available_uids = set(self.context.all_particles_uids)
        for qty in [
            *self.partQuantities,
            *self.spaceTimeHeatmaps,
            *self.movies1D,
            *self.movies2D,
        ]:
            if isinstance(qty.uids, list) and len(qty.uids) > 0:
                missing_uids = set(qty.uids) - available_uids
                if len(missing_uids) > 0:
                    raise Exception(
                        f"One or more requested particle uids do not exist: {sorted(missing_uids)}"
                    )

        partInfo = self.context.outputTypes_info["particles"]
        globalInfo = self.context.outputTypes_info["vtk"]
        if len(self.partQuantities) > 0 or self.particles_requested:
            if not partInfo.status:
                raise Exception(
                    f"Particle quantities were requested, but no part*.vtk files were found at {partInfo.files}"
                )

        if (
            len(self.spaceTimeHeatmaps) > 0
            or len(self.movies1D)
            or len(self.movies2D) > 0
        ):
            if not globalInfo.status:
                raise Exception(
                    f"Global quantities were requested, but no data*.vtk files were found at {globalInfo.files}"
                )

        LOG("Quantities to compute:")
        LOG(f"{'LineMovie1D':>20}: {len(self.movies1D)}")
        LOG(f"{'MapMovie2D':>20}: {len(self.movies2D)}")
        LOG(f"{'SpaceTimeHeatmap':>20}: {len(self.spaceTimeHeatmaps)}")
        LOG(f"{'PartQuantity':>20}: {len(self.partQuantities)}")

    def run(self):
        """
        Pray.
        """
        self._check_everything_alright()

        # -om -> Only renders Movie
        if self.userArgs.onlyMovie:
            self.renderer.render_movie()
            return

        vtktimes = None

        # Gather particles data
        if len(self.partQuantities) > 0:
            with Pool(self.userArgs.jobs) as pool:
                particles_result = pool.starmap(
                    self.processor.get_quantities,
                    zip(self.partList, repeat(self.partQuantities)),
                )

            nb_vtktimes = len(particles_result)
            vtktimes = [particles_result[i][0] for i in range(nb_vtktimes)]
            if len(vtktimes) > 1:
                t_smooth = np.linspace(min(vtktimes), max(vtktimes), 10000)
            else:
                t_smooth = np.array(vtktimes)

            for qty in self.partQuantities:
                values = np.array(
                    [particles_result[i][qty.index] for i in range(nb_vtktimes)]
                )
                qty.set_data(points=vtktimes, values=values)

                if qty.ref_function is not None:
                    try:
                        predicted_values = qty.ref_function(t_smooth)
                        qty.set_ref_data(t_smooth, predicted_values)
                    except Exception as e:
                        LOG(
                            f"Warning: Failed to compute ref_function for {qty.key}. Error: {e}"
                        )
            if self.particles_requested and len(self.context.active_directions) >= 2:
                # cartesian for pcolormesh
                self.processor.partsInfo.parts_Z.set_data(
                    *tools.convertGrid_toXZ(
                        self.processor.partsInfo.parts_X1.values,
                        self.processor.partsInfo.parts_X2.values,
                        self.context.geometry,
                    )
                )

        # gather spacetime data
        if len(self.spaceTimeHeatmaps) > 0:
            with Pool(self.userArgs.jobs) as pool:
                spat_results = pool.starmap(
                    self.processor.get_quantities,
                    zip(self.vtkList, repeat(self.spaceTimeHeatmaps)),
                )

            nb_vtktimes = len(spat_results)
            vtktimes = [spat_results[i][0] for i in range(nb_vtktimes)]

            for qty in self.spaceTimeHeatmaps:
                values = np.array(
                    [spat_results[i][qty.index] for i in range(nb_vtktimes)]
                )
                qty.set_data(points=self.processor.gridInfo.X1Line, values=values)

                if qty.ref_function is not None:
                    t_array = np.array(vtktimes)
                    if len(t_array) > 1:
                        t_smooth = np.linspace(t_array.min(), t_array.max(), 500)
                        qty.set_ref_data(t_smooth, qty.ref_function(t_smooth))
        if vtktimes is not None:
            self.processor.set_vtktimes(vtktimes)

        # delegate the render of all this stuff to the Renderer
        self.renderer.set_infos(self.processor.gridInfo, self.processor.partsInfo)
        self.renderer.render()

    def _name_frames(self):
        context = self.context

        self.slice1_list = context.get_slice1_vtkFiles()
        self.vtkList = context.get_global_vtkFiles()
        self.partList = context.get_particles_vtkFiles()

    def _apply_config(self):
        if self.userArgs.onlyMovie or self.userArgs.onlyAnalysis:
            return

        all_fields = [v.key for v in self.movies2D]
        config = self.context.config
        LOG(f"config.json file requested: {config}")
        all_bounds = {}

        if self.userArgs.noBounds:
            LOG("Bounds computation discarded.")
            return

        LOG("Computing bounds, please wait...")
        fields_tobound = [
            key
            for key in all_fields
            if key not in config or "bounds" not in config[key]
        ]

        if len(fields_tobound) > 0:
            bound_list = self.slice1_list if len(self.slice1_list) > 0 else self.vtkList
            computed_bounds = self._get_bounds(
                bound_list[min(len(bound_list), 5) :],
                fields_tobound,
            )
            LOG("Fields to bound: ", fields_tobound)
            [LOG(f"{key}: {computed_bounds[key]}") for key in computed_bounds]
            LOG("Bounds computed")

        else:
            LOG("All fields are already bounded in config")

        all_movies = [*self.movies1D, *self.movies2D]
        for qtyInfo in all_movies:
            qty_key = qtyInfo.key
            AVAILABLE_KWARGS = [
                "bounds",
                "symbol",
                "title",
                "style_kwargs",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "xscale",
                "yscale",
                "norm",
            ]
            if qty_key in config:
                for key in config[qty_key]:
                    if key in AVAILABLE_KWARGS:
                        setattr(qtyInfo, key, config[qty_key][key])

            if qty_key in all_bounds and not self.userArgs.noBounds:
                qtyInfo.set_bounds(all_bounds[qty_key])
            elif self.userArgs.noBounds:
                qtyInfo.set_bounds([None, None])

        LOG("Final Bounds:")
        for qtyInfo in all_movies:
            LOG(f"{qtyInfo.key:>10} {qtyInfo.bounds}")

    def _get_bounds(self, vtkList, fields_keys):
        """
        Get the bounds (min, max) of all given fields. I recommend not passing the entire vtkList but rather vtkList[1:] to discard the first output(s ?).

        vtkList    List of dump file paths
        field       Field

        returns
        dict where dict[field] = (min, max)
        """
        fieldskeys_indexes = {}
        for i, key in enumerate(fields_keys):
            fieldskeys_indexes[key] = i

        with Pool(self.userArgs.jobs) as pool:
            computed_bounds = pool.map(
                self._get_bounds_indiv,
                [[vtk, fieldskeys_indexes] for vtk in vtkList],
            )
        computed_bounds = np.array(computed_bounds)
        bounds = {}
        if len(computed_bounds) == 0:
            return bounds
        for key in fields_keys:
            i = fieldskeys_indexes[key]
            bounds[key] = (
                np.nanmin(computed_bounds[:, i, 0]),
                np.nanmax(computed_bounds[:, i, 1]),
            )
        return bounds

    def _get_bounds_indiv(self, args):
        """
        args (list[2]) must have two components:
            first:   vtk_path (str)
            second:   fields_indexes (dict) where fields_indexes[field] = index
        """
        vtk_path = args[0]
        fieldskeys_indexes = args[1]
        V = readVTK(vtk_path)
        self.processor.process(V)
        bounds = np.empty((len(fieldskeys_indexes), 2))
        for field in fieldskeys_indexes.keys():
            data = V.data[field]
            index = fieldskeys_indexes[field]
            bounds[index, 0] = np.nanmin(data)
            bounds[index, 1] = np.nanmax(data)
        return bounds


def _get_args():
    """Builds the default command-line argument parser for Idefix2Python."""
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

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of CPU cores to use",
    )

    parser.add_argument(
        "-u",
        "--until",
        type=lambda s: int(s) if s.isdigit() else float(s),
        default=1,
        help="To read only a part of the data. float between 0 and 1 is interpreted as a fraction, int as an output number, and a float > 1 as a time.",
        dest="until",
    )

    parser.add_argument(
        "-e",
        "--every",
        type=int,
        default=1,
        help="Read every Nth output file (N>=1). For example, -e 2 reads every second file.",
    )

    args = parser.parse_args()
    if args.doOnlyFrames is None:
        args.doOnlyFrames = False
    elif len(args.doOnlyFrames) == 0:
        args.doOnlyFrames = [0]

    return args
