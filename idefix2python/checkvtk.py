# %%
from vtk_io import readVTK
import numpy as np
import matplotlib.pyplot as plt

# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/Drift_Tau/vtks/data.0000.vtk"
path = "/home/dp316/dp316/dc-fang1/idefix/test/MHD/AmbipolarWind/data.0000.vtk"

data = readVTK(path).data
# print(vtk.geometry)

# print(np.average(vtk.data["Dust0_RHO"]))
# print(vtk.x)

for key in data:
    print(key)

# import glob

# for path in glob.glob(
#     "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/Drift_Tau/vtks/*.vtk"
# ):
#     print(path)
#     vtk = readVTK(path)
#     print(np.min(vtk.x))
#     # plt.plot(vtk.x)
#     # plt.savefig("test.png")

# v = readVTK(
#     "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/Drift_Tau/vtks/data.0000.vtk"
# )
# plt.plot(v.x, v.data["RHO"][:, :, 0])


# %%
