import numpy as np
import json
from itertools import zip_longest
import inifix


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


def get_streamplot_data(
    X1, X2, Ux_data, Uy_data, geometry, zoom=None, resolution=200, method="linear"
):
    """
    X,Y must be Lines, Ux and Uy are the data points.
    Returns the 1d arrays x_coords and y-coords, with the interpolated values Ux_uni, Uy_uni (2D arrays) that can go directly in streamplot
    """
    from scipy.interpolate import RegularGridInterpolator

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
