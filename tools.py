import os

import numpy as np
import json


def process_configs(config_path):
    with open(config_path) as f:
        configs = json.load(f)
        for runName in configs:
            if "copy" in configs[runName]:
                configs[runName] = configs[configs[runName]["copy"]]
    return configs


def dat_to_dict(path, end=1):
    d = {}
    with open(path) as f:
        lines = f.readlines()
        lastindex = int(len(lines) * end)
        keys = lines[0].split()
        for key in keys:
            d[key] = np.array([], dtype=np.float64)
        for line in lines[1:lastindex]:
            vals = line.split()
            for ii in range(len(keys)):
                d[keys[ii]] = np.append(d[keys[ii]], [float(vals[ii])])
    return d


def fmt(x, pos):
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


def formatInputs(iniPath):
    with open(iniPath) as ini:
        lines = ini.readlines()
        indexes_to_format = []

        # Grid part
        istart = lines.index("[Grid]\n") + 1
        iend = lines.index("[TimeIntegrator]\n")
        indexes_to_format += [*range(istart, iend)]
        # Setup part
        istart = lines.index("[Setup]\n") + 1
        iend = lines.index("[Output]\n")
        indexes_to_format += [*range(istart, iend)]
        text = ""
        for i in indexes_to_format:
            line = lines[i]
            line_split = line.split()
            if len(line_split) > 1 and line_split[0] != "#":
                text += f"{line_split[0]:>25} {' '.join(line_split[1:]):<10}"
                text += "\n"
    return text


def annotateInputs(axs, text):
    axs.flat[1].annotate(
        text,
        xy=(0.1, 0.9),
        xycoords="figure fraction",
        verticalalignment="center",
        horizontalalignment="left",
        family="monospace",
        fontsize=10,
    )


def fit(X, Y, deg, start=0, end=1):
    index_start = int(start * (len(X) - 1))
    index_end = int(end * (len(X) - 1))
    params, cov = np.polyfit(
        X[index_start:index_end], Y[index_start:index_end], deg=deg, cov=True
    )
    return params, np.diag(cov)


def divide_discardingNullDenominator(a, b):
    """
    Returns a/b but with None wherever b=0
    """
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.abs(b) > 1e-10)


def applyOperation_discardingNone(op, array):
    mask = (array != np.nan) & (array != 0)
    output = np.full(array.shape, np.nan)
    valid_data = array[mask].astype(float)
    output[mask] = op(valid_data)

    return output
    # return op(array, out=np.full(array.shape, None), where=array != None)


def movie(pattern_png, movie_path, fps=10):
    import ffmpeg

    # pattern_png = f"{folder_path}/*.png"
    ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
        movie_path,
        vcodec="libx264",
        crf=18,
        preset="medium",
        r=fps,
        pix_fmt="yuv420p",
        movflags="faststart",
    ).overwrite_output().run()
    print(f"[OK] {movie_path}")


def RequirePath(path, dir_or_file=None):
    if dir_or_file is None:
        if not os.path.exists(path):
            raise Exception
    elif dir_or_file == "dir":
        if not os.path.isdir(path):
            raise Exception
    elif dir_or_file == "file":
        if not os.path.isfile(path):
            raise Exception
    else:
        raise Exception("wrong value for dir_or_file")


def convertGrid_toXZ(X1, X2, geometry):
    if geometry == "cartesian":
        return X1, X2
    elif geometry == "cylindric":
        return X1, X2
    elif geometry == "polar":
        raise NotImplementedError("POLAR geometry not implemented yet")
    elif geometry == "spherical":
        return X1 * np.sin(X2), X1 * np.cos(X2)


def convertVector_toXZ(uX1, uX2, X1, X2, geometry):
    if geometry == "cartesian":
        return uX1, uX2
    elif geometry == "cylindric":
        return uX1, uX2
    elif geometry == "polar":
        raise NotImplementedError("POLAR geometry not implemented yet")
    elif geometry == "spherical":
        Theta = X2
        return np.sin(Theta) * uX1 + np.cos(Theta) * uX2, np.cos(Theta) * uX1 - np.sin(
            Theta
        ) * uX2
