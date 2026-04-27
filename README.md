# Idefix2Python

`Idefix2Python` is a Python post-processing and visualization pipeline designed for **Idefix** (https://github.com/idefix-code/idefix) simulation outputs. It is currently designed for MHD and Dust (Pressureless fluid or Lagrangian particles) simulations. 

## Usage
```bash
git clone https://github.com/DavidFang03/idefix2python.git
cd idefix2python
pip install .
```

The minimal example below shows $\rho(x,y)$ in a movie, if your simulation output is 2D.
```python
from idefix2python import RunContext, Pipeline, SpaceTimeHeatmap, MapMovie2D

ctx = RunContext(runName="test_run")

maps = [MapMovie2D("RHO", r"$\rho$")]

pipe = Pipeline(ctx, movies2D=maps)
pipe.run()
```
Check the docs for more examples and explore the other features: https://davidfang03.github.io/idefix2python/

## Supported quantity types

Users define what to plot by passing lists of "Quantity" objects to the Pipeline:

*   **`MapMovie2D`**: For $f(x, z, t)$ fields that will be rendered as a heatmap (pcolormesh) animation. Supports:
    *   `streamlines`: Vector overlays (e.g., velocity fields).
    *   `contours`: draw contour lines over the pcolormesh. (e.g., density levels).
    *   `compute`: Custom functions to calculate new fields on the fly.
*   **`LineMovie1D`**: For $f(x,t)$ fields, renders a line plot that updates every frame. (e.g., radial profile over time).
*   **`SpaceTimeHeatmap`**: For $f(x,t)$ fields, renders a space-time heatmap. Supports `ref_function` to overlay analytical trajectories.
*   **`PartQuantity`**: Tracks Lagrangian particle properties (like `PART_X1`) over time. 

`SpaceTimeHeatmap` and `PartQuantity` supports `ref_function` to overlay analytical functions.
One can also plot a particle quantity over a spacetime heatmap with `trace_over`



## Command line options

| Option | Argument | Description |
| :--- | :--- | :--- |
| `-j`, `--jobs` | `int` | Number of CPUs for parallel rendering (Default: 1). |
| `-z`, `--zoom` | `float` | Crops the plots to a specific radius ($r < \mathrm{zoom}$). |
| `-f`, `--frame` | `int...` | Renders only specific frame indices (e.g., `-f 0 10 -1`). |
| `--no-bounds` | Flag | Ignores `config.json`. User expects colobar to be different at each frame and to match local data. |
| `-om` | Flag | Only movie: Skips everything and only renders the movie on existing frames. |
| `-u`, `--until` | `float` or `int` | To read only a part of the data. `float` between 0 and 1 is interpreted as a fraction, `int` as an output number, and a `float` > 1 as a time. |
| `-e`, `--every` | `int` | Read every Nth output file (N>=1) . For example `-e 2` will read every second file. |


## Config file (`config.json`)

To ensure constant colorbars across the movies, the user can define fixed bounds in a JSON file. The pipeline will automatically apply these unless `--no-bounds` is used.

```json
{
    "RHO": {
        "bounds": [1e-3, 10],
        "cmap": "viridis",
        "norm": "log"
    }
}
```

## Architecture

The module is built around 5 components:

+  **`RunContext`**: Handles data location and directory creation. Detects simulation geometry, dimensions, and available fields.
+  **`Quantities`**: Defines all the different types of data and ways to visualize them.
+  **`PhysicsProcessor`**: Performs mathematical transformations. Handles grid conversions (e.g., converting internal coordinates to Cartesian $x, z$ for plotting), applies zooms, and computes derived quantities.
+  **`SliceRenderer`**: Matplotlib engine. Manages multi-panel layouts, colorbars (Log, Linear, TwoSlope), streamlines, contours.
+  **`Pipeline`**: Coordinates the detection, processing, and rendering of the simulation data. Computes the bounds and distributes with `multiprocessing`.

## TODOs

* **Lagrangian Dust**: Plot particles positions on the 2D heatmaps.
* Better colorbar and dynamic layout
* More flexibility on plot parameters (linestyle, color, etc...)
* Reintroduce `timevol.dat` (timevol) for global quantities.

### Not a priority
* Support multiple pipelines
* Support mixed outputs (e.g `data*.vtk` + `slice1*.vtk`)
* Add a `discard` option to replace non-physical values (e.g., $<0$) with `NaN`

### Not planned
*   Planet rendering/trajectories
*   Non-XZ planes
*   Simulations with less than 3 components
*   Support simultaneous rendering of global 1D and 2D slice outputs