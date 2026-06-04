import numpy as np
import json
from itertools import zip_longest


def LOG(*args):
    print(*args)


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
    """
    Formats the .ini file into a decent dict
    """
    with open(iniPath) as ini:
        content = ini.read()

    sections = {}
    current_section = None

    MAX_VAL_LEN = 22

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            sections[current_section] = []
        elif current_section:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                val = " ".join(parts[1:])
                val = "".join(val.split("#")[0])
                if len(val) > MAX_VAL_LEN:
                    val = "..." + val[MAX_VAL_LEN - 3 :]
                sections[current_section].append(f"{key:<14} {val}")

    return {k: "\n".join(v) for k, v in sections.items() if v}


def annotateInputs(fig, ini_dict, padding_top=0.0):
    """
    Writes text on the `fig` with distinctive sections.
    """
    if ini_dict == {}:
        return

    COL_WIDTH = 42
    COLS_NB = 3

    all_panels = []
    keys = list(ini_dict.keys())
    table = [keys[i : i + COLS_NB] for i in range(0, len(keys), COLS_NB)]

    for section_group in table:
        formatted_columns = [
            [f"[{name}]", *ini_dict[name].split("\n")]
            for name in section_group
            if name in ini_dict
        ]

        if not formatted_columns:
            continue

        printable_rows = [
            "".join(f"{section_line:<{COL_WIDTH}}" for section_line in horizontal_slice)
            for horizontal_slice in zip_longest(*formatted_columns, fillvalue="")
        ]

        text_panel = "\n".join(printable_rows)
        all_panels.append(text_panel)

    final_display_string = "\n\n\n".join(all_panels)

    total_lines = final_display_string.count("\n") + 1
    header_space = total_lines * 0.014
    margin_top = header_space + padding_top

    fig.text(
        0.55,
        1 - margin_top,
        final_display_string,
        family="monospace",
        fontsize=7,
        va="bottom",
        ha="center",
        usetex=False,
    )

    fig.subplots_adjust(top=1.0 - margin_top - 0.05)


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

    print(movie_path, "from", pattern_png)
    ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).filter(
        "scale",
        3840,
        "-2",  # TODO More flexible
    ).output(
        str(movie_path),
        vcodec="libx264",
        crf=18,
        preset="medium",
        r=fps,
        pix_fmt="yuv420p",
        movflags="faststart",
    ).overwrite_output().run()
    print(f"[OK] {movie_path}")


def convertLines_toXZgrid(X1, X2, X3, geometry):
    if geometry == "cartesian":
        return np.meshgrid(X1, X2)
    elif geometry == "cylindric":
        return np.meshgrid(X1, X2)
    elif geometry == "polar":
        return np.meshgrid(X1, X3)
    elif geometry == "spherical":
        grid = np.meshgrid(X1, X2)
        return grid[0] * np.sin(grid[1]), grid[0] * np.cos(grid[1])


def convertGrid_toXZ(X1, X2, geometry):
    if geometry == "cartesian":
        return X1, X2
    elif geometry == "cylindric":
        return X1, X2
    elif geometry == "polar":
        raise NotImplementedError("POLAR geometry not implemented yet")
    elif geometry == "spherical":
        return X1 * np.sin(X2), X1 * np.cos(X2)


def get_Position(file, geometry, direction):
    match geometry:
        case "cartesian":
            positions = [file.x, file.y, file.z]
        case "polar":
            positions = [file.x, file.y, file.z]
        case "cylindrical":
            positions = [file.r, file.z, None]
        case "spherical":
            positions = [file.r, file.theta, file.phi]
    return positions[direction]


def get_PositionL(file, geometry, direction):
    match geometry:
        case "cartesian":
            positions = [file.xl, file.yl, file.zl]
        case "polar":
            positions = [file.xl, file.yl, file.zl]
        case "cylindrical":
            positions = [file.rl, file.zl, None]
        case "spherical":
            positions = [file.rl, file.thetal, file.phil]
    return positions[direction]


def get_streamline_name(u_key1):
    """
    Determines a nice LaTeX name for the streamline based on the field key.
    """
    if u_key1 == "VX1":
        return r"$\mathbf{u}_\mathrm{p}$"
    elif u_key1 == "BX1":
        return r"$\mathbf{B}_\mathrm{p}$"
    elif u_key1.startswith("Dust") and "VX" in u_key1:
        dust_id = u_key1.split("_")[0]  # E.g. 'Dust0'
        return rf"$\mathbf{{v}}_{{p, \mathrm{{{dust_id}}}}}$"
    else:
        return str(u_key1).replace("1", "p")
