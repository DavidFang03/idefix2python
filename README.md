# idefix2python

The module is divided in 4 distinctive parts

- `Context` to manage the location of the Idefix outputs. Also prepare the ground for the future rendering. Behaviour depends only on the location of the Idefix outputs.

- `Processor` to process the data depending of the geometry, dimensions, and possible zoom or bounds.

- `Renderer` for all the matplotlib stuff

- `Pipeline` to make everything work together.
Will also determine the name of the renders depending on whether `--zoom` or `--no-bounds` has been requested.

Some minors classes include
- `Quantity1D`, `Quantity2D` to store the information of each requested quantity.
The differences are:
    - `Quantity1D` is a field that is function of one space dimension. Will be plotted as a timeline with pcolormesh. One can also as some reference points, like an analytical function.
    -  `Quantity2D` is a field that is function of two space dimension. Will be plotted as pcolormesh for each frame. Streamline can also be included.


Some runtime options can be requested
- `--zoom` if one wants to only plot the quantities on a subdomain.
- `--no-bounds` if one wants to ignore any `config.json` file for the bounds.
- `--frame` if one wants to only render some frames
- `-om`: if the pipeline has already rendered once and one wants to renders the movie only.
- `-oa`: if the pipeline has already rendered once and one wants to renders the analysis only. (deprecated?)


TODOs:
- Lagrangian dust support.
- Reintroduce the .dat (`analysis`)
- Currently the code doesn't support different types of outputs at the same time, for example 2D global and 1D slice.
- Add an option `discard` that replace some values (e.g <0) by np.nan

Not important
- For now the program loops a first time to compute the bounds, then a second time to collect all the TimeSeries and will loop again to render the 2D plots.

Not planned:
- Planet support
- Different plane than (x,z)
- Number of components < 3