# %%
from vtk_io import readVTK
import numpy as np
import matplotlib.pyplot as plt
import glob
# # path = "/home/dp316/dp316/dc-fang1/IdefixRuns/DriftSettling/outputs/DS_test/vtks/data.0000.vtk"
# path = "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/Drift_l_Tau/vtks/part.0000.vtk"
# vtk = readVTK(path)
# data = vtk.data


# for key in data:
#     print(key)
#     # print(data[key])

# print(vtk.geometry)

# print(np.average(vtk.data["Dust0_RHO"]))
# print(vtk.x)

# for key in data:
#     print(key)

# import glob

# for path in glob.glob(
#     "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/Drift_Tau/vtks/*.vtk"
# ):
#     print(path)
#     vtk = readVTK(path)
#     print(np.min(vtk.x))
#     # plt.plot(vtk.x)
#     # plt.savefig("test.png")

# # tu-c0r0n75
v = readVTK(
    "/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/outputs/DriftL_Tau/vtks/data.0001.vtk"
)
print(v.data.keys())
print(np.shape(v.data["RHO"][:, :, 0]))
plt.plot(v.x, np.transpose(v.data["RHO"][:, 0, 0]))
# plt.plot(v.x, 0.05 * v.x ** (-0.5))
# # tu-c0r0n93
# for proc in range(4):
#     # for npy in glob.glob("/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup/*.npy"):
#     npy = f"/home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup/debugRHO_proc{proc}.npy"
#     data = np.load(npy)
#     print(data)
#     plt.plot(data[0, 0, :])


# tu-c0r0n75
# v = readVTK("/home/dp316/dp316/dc-fang1/myidefix/test/HD/FargoPlanet/data.0000.vtk")
# x, y = np.meshgrid(v.x, v.y)
# print(np.shape(v.data["RHO"][:, :, 0]))
# plt.pcolormesh(x, y, np.transpose(v.data["RHO"][:, :, 0]))
# plt.plot(v.x, np.transpose(v.data["RHO"][:, 0, 0]))

# %%
