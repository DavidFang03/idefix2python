# Idefix2Python

`Idefix2Python` is a high-level post-processing and visualization pipeline designed for **Idefix** (https://github.com/idefix-code/idefix) simulation outputs. It is currently designed for MHD and Dust (Pressureless fluid or Lagrangian particles) simulations. 

## Architecture

The module is built around 4 components:

+  **`RunContext`**: Handles IO, directory creation, and data discovery. Detects simulation geometry, dimensions, and available fields.
+  **`PhysicsProcessor`**: Performs mathematical transformations. Handles grid conversions (e.g., converting internal coordinates to Cartesian $x, z$ for plotting), applies zooms, and computes derived quantities.
+  **`SliceRenderer`**: Matplotlib engine. Manages multi-panel layouts, colorbars (Log, Linear, TwoSlope), streamlines, contours.
+  **`Pipeline`**: The coordinator:
    *   Pre-calculates global bounds across all frames.
    *   Distributes rendering tasks across multiple CPU cores.
    *   Manages the temporal evolution of SpaceTime heatmaps and particle data.

## Supported quantity types

Users define what to plot by passing lists of "Quantity" objects to the Pipeline:

*   **`MapMovie2D`**: For $f(x, z, t)$ fields, 2D pcolormesh plots generated for every frame. Supports:
    *   `streamlines`: Vector overlays (e.g., velocity fields).
    *   `contours`: Scalar overlays (e.g., density levels).
    *   `compute`: Custom functions to calculate new fields on the fly.
*   **`LineMovie1D`**: For $f(x,t)$ fields, 1D plot generated for every frame (e.g., radial profile over time).
*   **`SpaceTimeHeatmap`**: For $f(x,t)$ fields, generates a space-time heapmap. Supports `ref_function` to overlay analytical trajectories.
*   **`PartQuantity`**: Tracks Lagrangian particle properties (like `PART_X1`) over time. Supports `ref_function` to overlay analytical functions.

## Usage

```python
from Idefix2Python import RunContext, Pipeline, SpaceTimeHeatmap, MapMovie2D

# 1. Setup Context
ctx = RunContext(projectPath="./my_sim", runName="test_run", configPath="./config.json")

# 2. Define Plots
heatmaps = [SpaceTimeHeatmap("Dust0_RHO", r"$\rho_d$", plot_coords=[0,0], ref_function=my_theory)]
maps = [MapMovie2D("RHO", r"$\rho$", plot_coords=[0,0], streamlines=["VX1", "VX2"])]

# 3. Run Pipeline
pipe = Pipeline(ctx, spaceTimeHeatmaps=heatmaps, movies2D=maps)
pipe.run()
```

## Command line options

| Option | Argument | Description |
| :--- | :--- | :--- |
| `-j`, `--jobs` | `int` | Number of CPUs for parallel rendering (Default: 1). |
| `-z`, `--zoom` | `float` | Crops the plots to a specific radius ($r < \mathrm{zoom}$). |
| `-f`, `--frame` | `int...` | Renders only specific frame indices (e.g., `-f 0 10 -1`). |
| `--no-bounds` | Flag | Ignores `config.json`. User expects colobar to be different at each frame and to match local data. |
| `-om` | Flag | Only movie: Skips everything and only renders the movie on existing frames. |


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

## TODOs

* **Lagrangian Dust**: Plot particles positions on the 2D heatmaps.
* More flexibility on plot parameters (linestyle, color, etc...)
* Reintroduce `timevol.dat` (timevol) for global quantities.

### Not prority
* Support mixed outputs (e.g `data*.vtk` + `slice1*.vtk`)
* Add a `discard` option to replace non-physical values (e.g., $<0$) with `NaN`

### Not planned
*   Planet rendering/trajectories
*   Non-XZ planes
*   Simulations with less than 3 components
*   Support simultaneous rendering of global 1D and 2D slice outputs