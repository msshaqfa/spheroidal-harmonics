#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:39:24 2023

@author: mahmoud
"""

import numpy as np
from scipy.interpolate import interp1d

import igl

from plot_helper import *
from spheroid_coordinates import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def surface_cumulator(t, u, coords):
    """
    Parameters
    ----------
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    u : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t, u pair.

    Returns
    -------
    t, u : arrays
        As above.
    cum_S_t : array
        Cumulative surface area on [t[0], t], all u.
    cum_S_u : array
        Cumulative surface area on all t, [u[0], u].

    Evaluates the cumulative surface area at each coordinate.
    
    Author: Max Kapur, https://github.com/maxkapur/param_tools/tree/master
    """

    if np.all(t) is None:
        t, _ = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))
    if np.all(u) is None:
        _, u = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))

    assert t.shape == u.shape == coords.shape[1:], \
        "Need same number of parameters as coordinates"
    delta_t_temp = np.diff(coords, axis=2)
    delta_u_temp = np.diff(coords, axis=1)

    # Pad with zeros so that small rand_S can still be interpd
    delta_t = np.zeros(coords.shape)
    delta_u = np.zeros(coords.shape)

    delta_t[:coords.shape[0], :coords.shape[1], 1:coords.shape[2]] = delta_t_temp
    delta_u[:coords.shape[0], 1:coords.shape[1], :coords.shape[2]] = delta_u_temp

    # Area of each parallelogram
    delta_S = np.linalg.norm(np.cross(delta_t, delta_u, 0, 0), axis=2)

    cum_S_t = np.cumsum(delta_S.sum(axis=0))
    cum_S_u = np.cumsum(delta_S.sum(axis=1))
    return t, u, cum_S_t, cum_S_u



def polyhedron_spheroid(f, zeta, spheroid_type, theta_sections, phi_sections, 
                        refine_poles =  True, ref_cycles = 2, plot = False, 
                        save_name = "spheroid_base_mesh.stl"):
    if spheroid_type == "prolate" or spheroid_type == "sphere":
        domain_theta = [0, np.pi/2]
        coord_func = prolate2cart
    elif spheroid_type == "oblate":
        domain_theta = [0, np.pi/2]
        coord_func = oblate2cart
    
    domain_phi = [0, 2* np.pi]
    
    # Surface data
    data_theta, data_phi = np.meshgrid(np.linspace(*domain_theta, 100), np.linspace(*domain_phi, 100))
    x_, y_, z_ = coord_func(data_theta, data_phi, f, zeta)
    data_z = np.array([x_, y_, z_])
    
    # Cumulation data
    cum_S_t, cum_S_u = surface_cumulator(data_theta, data_phi, data_z)[2:]
    
    # Choose regularly spaced intervals of area and invert to parameter space
    points_S_t, points_S_u = np.meshgrid(np.linspace(0,cum_S_t[-1], theta_sections),
                                         np.linspace(0,cum_S_u[-1], phi_sections))
    
    # Evaluate the function over received parameters
    points_t = interp1d(cum_S_t, data_theta[0,:])(points_S_t)
    points_u = interp1d(cum_S_u, data_phi[:,0])(points_S_u)
    
    # Refine near poles
    if refine_poles:
        for _ in range(ref_cycles):
            if spheroid_type == "prolate" or spheroid_type == "sphere":
                points_t = np.insert(points_t, 1, 0.5 * (points_t[0, 1] + points_t[0, 0]) , axis=1)
                points_u = np.insert(points_u, 0, points_u[:, 0] , axis=1) # It will be deleted with cleaning (trick evals)
            elif spheroid_type == "oblate":
                points_t = np.insert(points_t, -2, 0.5 * (points_t[0, -2] + points_t[0, -1]) , axis=1)
                points_u = np.insert(points_u, -1, points_u[:, -1] , axis=1) # It will be deleted with cleaning (trick evals)

    # Evaluate new points
    x2_, y2_, z2_ = coord_func(theta=points_t, phi=points_u, f=f, zeta=zeta)
    points_x = np.array([x2_, y2_, z2_])
    
    if plot:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(hspace=0, wspace=0.1, bottom=.2)
        ax = fig.add_subplot(121, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Function values', y=1.09, size=15)
        ax.plot_surface(*data_z, color='blue', ls="--", alpha=0.15)
        ax.scatter(*points_x, marker='+', alpha=.7, color='forestgreen')
        
        ax2 = fig.add_subplot(122)
        ax2.scatter(points_t, points_u, marker='+', alpha=.7, color='forestgreen')
        set_axes_equal(ax)
        ax2.set_title('Parameter space', y=1, size=15)
        ax2.set_xlabel('theta')
        ax2.set_ylabel('phi')
    
    # Creating triangulated mesh
    triang = tri.Triangulation(points_x[0].flatten(), points_x[1].flatten())
    top_verts = np.array([points_x[0].flatten(), points_x[1].flatten(), points_x[2].flatten()]).T
    top_faces = triang.triangles
    bottom_verts = np.array([points_x[0].flatten(), points_x[1].flatten(), -points_x[2].flatten()]).T
    
    # Flip the normals of the bottom side (CCW to CW)
    bottom_faces = triang.triangles
    bottom_faces2 = np.array([bottom_faces[:, 2], bottom_faces[:, 1], bottom_faces[:, 0]]).T
    
    # Merge faces
    merged_verts = np.concatenate((top_verts, bottom_verts))
    bottom_faces2 += len(top_verts)
    merged_faces = np.concatenate((top_faces, bottom_faces2))
    
    # Clean replicated verts
    [SV,SVI,SVJ,SF] = igl.remove_duplicate_vertices(merged_verts, merged_faces, 1e-12);
    F = SVJ[merged_faces]
    V = SV
    
    # Export and save mesh
    igl.write_triangle_mesh(save_name , V, F)
    
    return points_x, V, F

def generate_icosahedron(scale=1.0):
    # The golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [[-1, t, 0],
                [1, t, 0],
                [-1, -t, 0],
                [1, -t, 0],
                [0, -1, t],
                [0, 1, t],
                [0, -1, -t],
                [0, 1, -t],
                [t, 0, -1],
                [t, 0, 1],
                [-t, 0, -1],
                [-t, 0, 1]]
    faces = [[0, 11,  5],
             [0,  5,  1],
             [0,  1,  7],
             [0,  7, 10],
             [0, 10, 11],
             [2, 11, 10],
             [4,  5, 11],
             [9,  1,  5],
             [8,  7,  1],
             [6, 10,  7],
             [4,  9,  5],
             [9,  8,  1],
             [8,  6,  7],
             [6,  2, 10],
             [2,  4, 11],
             [3,  9,  4],
             [3,  4,  2],
             [3,  2,  6],
             [3,  6,  8],
             [3,  8,  9]]
    return scale * np.array(vertices), np.array(faces)

def plot_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')

    # Plot the faces
    poly3d = [[vertices[vert_id] for vert_id in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='b', alpha=0.3))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()

def loop_subdivide(vertices, faces, subdivisions=1):
    for _ in range(subdivisions):
        new_vertices = vertices.copy()
        new_faces = []

        for face in faces:
            v1, v2, v3 = face
            a = (vertices[v1] + vertices[v2]) / 2
            b = (vertices[v2] + vertices[v3]) / 2
            c = (vertices[v1] + vertices[v3]) / 2

            new_vertex_indices = len(new_vertices), len(new_vertices) + 1, len(new_vertices) + 2
            new_vertices = np.vstack((new_vertices, a, b, c))

            new_faces.extend([
                [v1, new_vertex_indices[0], new_vertex_indices[2]],
                [new_vertex_indices[0], v2, new_vertex_indices[1]],
                [new_vertex_indices[0], new_vertex_indices[1], new_vertex_indices[2]],
                [new_vertex_indices[2], new_vertex_indices[1], v3]
            ])

        # Normalize new vertices to the ellipsoid's surface
        norms = np.linalg.norm(new_vertices, axis=1, keepdims=True)
        new_vertices = new_vertices / norms

        vertices = new_vertices
        faces = new_faces

    return vertices, faces

def spheroidal_icosahedron(a, c, subdivisions = 2):
    vertices, faces = generate_icosahedron(scale=1.0)
    subdivided_vertices, subdivided_faces = loop_subdivide(vertices, faces, subdivisions)
    subdivided_vertices[:, 0] = subdivided_vertices[:, 0] * a
    subdivided_vertices[:, 1] = subdivided_vertices[:, 1] * a
    subdivided_vertices[:, 2] = subdivided_vertices[:, 2] * c
    return subdivided_vertices, subdivided_faces



if __name__ == "__main__":
    
    #spheroid_type = "oblate"
    spheroid_type = "prolate"
    theta_sections = 40
    phi_sections = 20
    f, zeta = 30, 0.5    
    output_pc = polyhedron_spheroid(f, zeta, spheroid_type, theta_sections, phi_sections, refine_poles =  True, plot = True)
    
    # Icosahedron
    # Generate and plot the subdivided icosahedron
    vertices, faces = generate_icosahedron(scale=1.0)
    subdivided_vertices, subdivided_faces = loop_subdivide(vertices, faces, subdivisions=2)
    
    print("Number of vertices:", len(subdivided_vertices))
    print("Number of faces:", len(subdivided_faces))
    plot_mesh(vertices, faces)
    
    print("Subdivided Icosahedron:")
    plot_mesh(subdivided_vertices, subdivided_faces)
    
    # Spheroidal icosahedron
    a, c = 1., 5.
    sph_verts, sph_face = spheroidal_icosahedron(a, c, subdivisions = 3)
    plot_mesh(sph_verts, sph_face)
    