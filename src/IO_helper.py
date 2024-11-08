#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 06:37:04 2023

@author: mahmoud
"""

# Mesh IO stuff (better than meshIO lib)
import numpy as np
import evtk, os, meshio
vtkhl = evtk.hl # to get unstructuredGridToVTK
vtk = evtk.vtk  # to get VtkTriangle, VtkGroup

def createIOFolder(FOLDER_NAME):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

def cleanGroupPVD(FILE_PATH):
    try:
        os.remove(FILE_PATH + ".pvd")
    except:
        pass

def cleanFileVTU(FILE_PATH):
    try:
        os.remove(FILE_PATH + ".vtu")
    except:
        pass

def saveFileVTU(file_name = "",
                vertices = [],
                connectivity = [], verts_data = {}, faces_data = None):
    # For exporting unstructured triangular meshes only
    cleanFileVTU(file_name)
    conn = connectivity.ravel()
    ctype = np.zeros(int(len(conn)/3))
    ctype.fill(vtk.VtkTriangle.tid)
    offset = np.zeros(len(ctype), dtype=np.int64)
    offset[0] = 3
    for i in range(1, len(offset)):
        offset[i] = offset[i -1] + 3
    offset = np.ascontiguousarray(offset)
    x = np.ascontiguousarray(vertices[:,0])
    y = np.ascontiguousarray(vertices[:,1])
    z = np.ascontiguousarray(vertices[:,2])
    vtkhl.unstructuredGridToVTK(file_name, x, y, z,
                                connectivity = conn, offsets = offset,
                                cell_types = ctype, cellData = faces_data,  pointData = verts_data)
    return;

def saveGroupPVD(file_name, data_path):
    vtus = []
    for file in os.listdir(data_path):
        if file.endswith(".vtu"):
            vtus.append(os.path.join(data_path, file))
    vtus = sorted(vtus)
    g = vtk.VtkGroup(file_name)
    for time, i in enumerate(vtus):
        g.addFile(filepath = i, sim_time = time)
    g.save()
    return;

def writeVTKMeshes(fname, nodes, faces, point_data ={}):
    """
    This function to write meshes and the associated data.
    e.g. of point_data dictionary: point_data = {"random": np.random.rand(node_coordinates[:, 0].shape[0])}
    VTK output data guide: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    """
    meshio.write_points_cells(fname,
                              nodes,
                              {"triangle": faces},
                              point_data = point_data,
                              binary = True)
    return;
    
def save_ply(filename, points):
    num_points = len(points)
    try:
        with open(filename, 'w') as file:
            # Write PLY header
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write("element vertex {}\n".format(num_points))
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")

            # Write point coordinates
            for point in points:
                file.write("{} {} {}\n".format(point[0], point[1], point[2]))
            print("True")
    except:
        print("False")
    return;

if __name__ == "__main__":
    pass
