2D fields
=========

This page demonstrates 2D visualization, including streamlines and contours.

The data is taken from the AmbipolarWind test setup (https://github.com/idefix-code/idefix/tree/master/test/MHD/AmbipolarWind)

.. tab-set::

   .. tab-item:: Automatic Bounds

      .. literalinclude:: ../../../examples/run_2D.py
         :language: python
         :linenos:
      
      .. video:: ../../../examples/data_test/videos/2D_test.mp4
         :width: 100%
         :autoplay:
         :loop:
         :muted:
         :caption: 2D_test.mp4

      
      The result is not very good because the pipeline automatically computed the bounds of the colorbar by looking at the minimum/maximum values all over the simulation. Also, you might want a different colormap.
      
      There are two ways to add bounds:

      * Passing ``vmin`` and ``vmax`` arguments to the quantities (see the :doc:`run_particles` page for an example).
      * Adding a ``config.json`` file.

   .. tab-item:: Fixed Bounds (config.json)

      .. literalinclude:: ../../../examples/run_2D_bounds.py
         :language: python
         :linenos:

      .. literalinclude:: ../../../examples/data_test/config.json
         :language: json
         :caption: config.json
         :linenos:
      
      .. video:: ../../../examples/data_test/videos/config_2D_test.mp4
         :width: 100%
         :autoplay:
         :loop:
         :muted:
         :caption: config_2D_test.mp4

