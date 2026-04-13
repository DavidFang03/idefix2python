from vtk_io import readVTK

path = "/home/dp316/dp316/dc-fang1/IdefixRuns/DriftSettling/outputs/DS_test/vtks/data.0000.vtk"
vtk = readVTK(path)
data = vtk.data


# for key in data:
#     print(key)
#     print(data[key])

print(vtk.geometry)

# for key in data:
#     print(key)
