1D fields
=========

For 1 space dimension fields :math:`f(x,t)` there are 2 ways to render:

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Movie 1D

      Each frame shows the field at a specific time :math:`f(x,t_i)`.

      .. literalinclude:: ../../../examples/run_1D_movie.py
         :language: python
         :linenos:

      .. video:: ../../../examples/data_test/videos/1D_test.mp4
         :width: 600
         :autoplay:
         :loop:
         :muted:

   .. grid-item-card:: Space-Time Heatmap

      A single figure showing the entire evolution of the field :math:`f(x,t)`.

      .. literalinclude:: ../../../examples/run_1D_heatmap.py
         :language: python
         :linenos:

      .. image:: ../../../examples/data_test/frames/1D_test/1D_test_spacetimeheatmap.png
         :width: 80%
         :align: center

Adding a reference line
-----------------------

On the heatmap, one can add a reference line (e.g., an analytical trajectory) by passing a ``ref_function`` to the ``SpaceTimeHeatmap`` object.

.. literalinclude:: ../../../examples/run_1D_heatmap_ref.py
   :language: python
   :linenos:

.. image:: ../../../examples/data_test/frames/1D_test_withref/1D_test_spacetimeheatmap.png
   :width: 80%
   :align: center