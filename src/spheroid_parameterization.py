#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:05:17 2023

@author: mahmoud

%*************************************************************************%
% All rights reserved (C) to the authors: Mahmoud S. M. SHAQFA,           %
%                                         and Wim M. van Rees             %
%                                                                         %
% M. Shaqfa Contact:                                                      %
% Department of Mechanical Engineering, Massachusetts Institute of        %
% Technology (MIT)                                                        %
% Cambridge, MA, USA                                                      %
%               Email: mshaqfa@mit.edu                                    %
%               Email: wvanrees@mit.edu                                   %
%                                                                         %
%*************************************************************************%
% This code includes implementations for:                                 %
%				- Spheroidal harmonics of closed genus-0 surfaces         %
% This code is part of the paper: "Spheroidal harmonics for generalizing  %
% the morphological decomposition of closed parametric surfaces"          %
%                                                                         %
%*************************************************************************%
% This library is free software; you can redistribute it and/or modify	  %
% it under the terms of the GNU Lesser General Public License as published%
% by the Free Software Foundation; either version 2.1 of the License, or  %
% (at your option) any later version.                                     %
%                                                                         %
% This library is distributed in the hope that it will be useful,         %
% but WITHOUT ANY WARRANTY; without even the implied warranty of          %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                    %
% See the GNU Lesser General Public License for more details.        	  %
% You should have received a copy of the GNU Lesser General Public License%
% along with this library; if not, write to the Free Software Foundation, %
% Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA       %
%*************************************************************************%
Authors of this file: Mahmoud S. M. Shaqfa
"""
import sys, warnings
import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from spheroid_coordinates import *
import igl
from IO_helper import *
from quads_fit import *
from surface_registration import *
from render_stl import *

class parametrization:
    def __init__(self, V, F, spheroid_type, f, zeta):
        self.V = V
        self.F = F
        self.spheroid_type = spheroid_type
        self.f = f
        self.zeta = zeta
        if self.spheroid_type == "prolate":
            self.cart2spheroid = cart2prolate
            self.cart2spheroid_nonlinear = cart2prolate2
            self.spheroid2cart = prolate2cart
            self.a, self.c = f * np.sinh(zeta), f * np.cosh(zeta)
        elif self.spheroid_type == "oblate":
            self.cart2spheroid = cart2oblate
            self.cart2spheroid_nonlinear = cart2oblate2
            self.spheroid2cart = oblate2cart
            self.a, self.c = f * np.cosh(zeta), f * np.sinh(zeta)
        elif self.spheroid_type == "sphere":
            self.cart2spheroid = cart2sph
            self.spheroid2cart = sph2cart
            self.a, self.c = 1., 1.
    
    def compute_radial_para(self):
        # Check convexity
        if len(ConvexHull(self.V).vertices) == len(self.V):
            convex_type = "convex"
        else:
            convex_type = "nonconvex"
            print("This surface is non-convex and might be non-star-shaped!")
        # Compute parametric coordiantes radially (linear)
        theta, phi = self.cart2spheroid(self.V[:, 0], self.V[:, 1], 
                                          self.V[:, 2], self.f, self.zeta)
        x_, y_, z_ = self.spheroid2cart(theta, phi, self.f, self.zeta)
        self.V_map_reg = np.array([x_, y_, z_]).T
        
        # Check if the surface is star-shaped
        convexity = is_triangles_orientation_consistent(self.V_map_reg, self.F)
        if not convexity:
            warnings.warn("The mapped surface is not star-shaped!",
                          UserWarning)
        return self.V_map_reg, theta, phi
    
    def compute_nonlinear_radial_para(self):
        # Check convexity
        if len(ConvexHull(self.V).vertices) == len(self.V):
            convex_type = "convex"
        else:
            convex_type = "nonconvex"
            print("This surface is non-convex and might be non-star-shaped too!")
        # Compute parametric coordiantes - nonlinear radial mapping
        theta, phi = self.cart2spheroid_nonlinear(self.V[:, 0], self.V[:, 1], 
                                          self.V[:, 2], self.f, self.zeta)
        x_, y_, z_ = self.spheroid2cart(theta, phi, self.f, self.zeta)
        self.V_map_reg = np.array([x_, y_, z_]).T
        
        # Check if the surface is star-shaped
        convexity = is_triangles_orientation_consistent(self.V_map_reg, self.F)
        if not convexity:
            warnings.warn("The mapped surface is not star-shaped!",
                          UserWarning)
        return self.V_map_reg, theta, phi
    
    
    def compute_cMCF_para(self, iterations, time_step, verts_limit = 5000,
                          nonlinear_proj=False, plotresults = False):
        '''
        Implementation of the conformalized Mean Curvature Flow (cMCF) method
        
        
        parameters:
            iterations: maximum number of iterations provided for the analysis
        
        output:
            V_map: #V x 3 numpy-type array for the mapped vertices.
            theta, phi: #V arrays for the spheroidal coordinates.
        
        '''
        self.iterations = iterations
        
        # Compute initial curvature
        # Initialize smoothing with base mesh
        U = self.V
        K_G = igl.gaussian_curvature(U, self.F)
        M = igl.massmatrix(U, self.F, 
                           igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.sparse.diags(1 / M.diagonal())
        Kn = Minv.dot(K_G)
        self.original_Kn = Kn
        
        # Compute Laplace-Beltrami operator: #V by #V
        L = igl.cotmatrix(U, self.F)
        
        # Diagonal per-triangle "mass matrix"
        dblA = igl.doublearea(U, self.F)
        original_area = dblA.sum()
        
        self.RMSE = np.zeros((iterations+1, ))
        self.evolution_hist = np.zeros([len(U), iterations, 3])
        self.area_evolution_hist = np.zeros([len(U), iterations, 3])
        
        # Random sampling for error quantification
        verts_limit = int(verts_limit)
        if len(self.V[:,1]) > verts_limit:
            _ip = np.random.permutation(len(self.V[:,1]))
            _idel = _ip[:int(verts_limit)]
        else:
            _idel = np.arange(len(self.V[:,1]))
        
        # For registering prolates case
        th_y = np.pi/2
        R_y = np.array([[np.cos(th_y), 0, np.sin(th_y)],
                        [0, 1, 0], 
                        [-np.sin(th_y), 0, np.cos(th_y)]]
                       )
        self.R_y = R_y
        
        for i in range(iterations):
            # Recompute just mass matrix on each step 
            # (metrics will be constant for conformal mapping)
            M = igl.massmatrix(U, self.F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
            # Solve (M-delta*L) U = M*U
            S = (M - time_step * L)
            U = igl.spsolve(S, M * U)
            
            # Compute centroid and subtract (important for numerics)
            dblA = igl.doublearea(U, self.F)
            area = 0.5 * dblA.sum()
            BC = igl.barycenter(U, self.F);
            centroid = np.zeros(BC[0].shape)
            
            # Normalize the centroid
            for i_ in range(len(BC)):
                centroid += (0.5 * dblA[i_] / area) * BC[i_]
            U -= centroid
            
            # Normalize to unit surface area (important for numerics)
            U /= np.sqrt(area)
            U2 = U * np.sqrt(original_area * 0.5) # Preserve scale export/compare
            self.evolution_hist[:, i, :] = U2
            area_per_face = np.column_stack( [ dblA * 0.5,
                                              np.zeros((len(dblA), 2)) ] )
            self.area_evolution_hist[:, i, :] = igl.average_onto_vertices(U, self.F,
                                                                          area_per_face)
            # Error quantification
            '''
            self.RMSE[i] = np.mean(np.sqrt(abs(1 - (spheroid_func(U2[:, 0], 
                             U2[:, 1], U2[:, 2], self.a, self.c) + 1)**2)))
            '''
            # Error quantification starategy 2
            # Register surface (if it fails at early stages only)
            try:
                _, R_U = register_surface(U2[_idel, :], verts_limit)
                U22 = (R_U.T @ U2.T).T
                # Fit spheroid
                C_sph, spheroid_type = fit_spheroids(U22[_idel, :])
                if spheroid_type == "prolate":
                    U22 = (R_y @ U22.T).T
            except:
                U22 = U2
                spheroid_type = "sphere"
                C_sph = np.ones(4)
                print("Failed registration!")
            
            # RMSE
            self.RMSE[i] = np.mean(  np.sqrt(abs(1 - (spheroid_func(U22[:, 0],
                                                               U22[:, 1], 
                                                               U22[:, 2], 
                                                               C_sph[0], 
                                                               C_sph[1]) + 1)**2)) )
            """
            if len(ConvexHull(U2).vertices) == len(U2):
                convex_type = "convex"
            else:
                convex_type = "nonconvex"
            print("Iteration {} / {}, and the current surface is {}".format(i+1,
                                                                            iterations,
                                                                            convex_type),
                  end='\r')
            """
            # Iterations counter
            print("Iteration {} / {}".format(i+1, iterations), end='\r')
        print("Final Surface area = {}".format(igl.doublearea(U2, self.F).sum() * 0.5))
        
        # Stats for the area evolution
        self.avg_area_evol = np.zeros(iterations)
        self.min_area_evol = np.zeros(iterations)
        self.max_area_evol = np.zeros(iterations)
        self.std_area_evol = np.zeros(iterations)
        for j in range(iterations):
            area_p_v = self.area_evolution_hist[:, j, 0]
            self.avg_area_evol[j] = area_p_v.mean()
            self.max_area_evol[j] = area_p_v.max()
            self.min_area_evol[j] = area_p_v.min()
            self.std_area_evol[j] = area_p_v.std()
        
        # Save and return best spheroidal fit
        optimal_idx = np.where(self.RMSE == 
                               self.RMSE[np.nonzero(self.RMSE)].min())[0][0]
        if optimal_idx == 0: optimal_idx = iterations-1 # solve sphere bug
        self.V_map = self.evolution_hist[:, optimal_idx, :]
        # Register surface (more accuracy)
        _, self.R_U = register_surface(self.V_map, 20000)
        self.V_map_reg = (R_U.T @ self.V_map.T).T
        # Fit spheroid
        self.C_sph, self.spheroid_type = fit_spheroids(self.V_map_reg)
        if spheroid_type == "prolate":
            self.V_map_reg = (R_y @ self.V_map_reg.T).T
        
        print("Optimal spheroid iteration: {}".format(optimal_idx))
        
        # Compute parametric coordiantes (approximate!)
        if self.spheroid_type == "prolate":
            self.cart2spheroid = cart2prolate
            if nonlinear_proj:
                self.cart2spheroid = cart2prolate2
            self.spheroid2cart = prolate2cart
        elif self.spheroid_type == "oblate":
            self.cart2spheroid = cart2oblate
            if nonlinear_proj:
                self.cart2spheroid = cart2oblate2
            self.spheroid2cart = oblate2cart
        theta, phi = self.cart2spheroid(self.V_map_reg[:, 0],
                                        self.V_map_reg[:, 1],
                                        self.V_map_reg[:, 2],
                                        self.C_sph[2],
                                        self.C_sph[3])
        
        # Check if the surface is star-shaped
        convexity = is_triangles_orientation_consistent(self.V_map_reg, self.F)
        if not convexity:
            warnings.warn("The mapped surface is not star-shaped!",
                          UserWarning)
        
        if plotresults:
            plt.loglog(np.arange(iterations), self.RMSE[:-1])
            plt.xlabel("Iterations")
            plt.ylabel("RMSE")
            plt.title("Spheroidal-shape RMSE error")
            plt.show()
        return self.V_map_reg, theta, phi
    
    def dump_data_cMCF(self, filenames_dict, renderstl = False):
        assert 'evolution_hist' in dir(self), "Please run the compute_cMCF_para method"
        
        # Dump files names
        timesteps_folder = filenames_dict["timesteps_folder"]
        timesteps_files = filenames_dict["timesteps_files"]
        group_folder = filenames_dict["group_folder"]
        group_file = filenames_dict["group_file"]
        
        # Mesh IO
        createIOFolder(timesteps_folder)
        saveFileVTU(timesteps_files, self.V, self.F,
                    verts_data = {"curvature": np.ascontiguousarray(self.original_Kn)})
        
        for i in range(self.iterations):
            U = self.evolution_hist[:, i, :]
            area_p_v = self.area_evolution_hist[:, i, 0]
            
            # Compute curvature evolution
            K_Gn = igl.gaussian_curvature(U, self.F)
            Mn = igl.massmatrix(U, self.F, igl.MASSMATRIX_TYPE_VORONOI)
            Minvn = sp.sparse.diags(1 / Mn.diagonal())
            Knn = Minvn.dot(K_Gn)
            
            # Dump time steps
            temp_file = "sim_" + str(i).zfill(6);
            saveFileVTU(timesteps_folder + "/" + temp_file, U, self.F,
                        verts_data = {"curvature": np.ascontiguousarray(Knn),
                                      "area_evolution": np.ascontiguousarray(area_p_v)})
            
            if renderstl:
                render_surfaces_gnuplot(fname = "_" + temp_file + ".stl",
                                        directory = timesteps_folder,
                                        V = U, F = self.F,
                                        extra_field = [])
            
            print("Saved {} / {} files.".format(i+1, self.iterations), end='\r')
        # Group dumped data
        saveGroupPVD(file_name = group_file,
                     data_path = timesteps_folder)
    
    def dump_final_parameterization(self, filename):
        # self.V_map_reg, self.F
        igl.write_triangle_mesh(filename, self.V_map_reg, self.F)
    
    @staticmethod
    def quantify_conformal_error():
        pass
    
    def __del__(self):
        pass
    
    def dump_data_cMCF(self, filenames_dict, renderstl = False):
        assert 'evolution_hist' in dir(self), "Please run the compute_cMCF_para method"
        
        # Dump files names
        timesteps_folder = filenames_dict["timesteps_folder"]
        timesteps_files = filenames_dict["timesteps_files"]
        group_folder = filenames_dict["group_folder"]
        group_file = filenames_dict["group_file"]
        
        # Mesh IO
        createIOFolder(timesteps_folder)
        saveFileVTU(timesteps_files, self.V, self.F,
                    verts_data = {"curvature": np.ascontiguousarray(self.original_Kn)})
        
        for i in range(self.iterations):
            U = self.evolution_hist[:, i, :]
            area_p_v = self.area_evolution_hist[:, i, 0]
            
            # Compute curvature evolution
            K_Gn = igl.gaussian_curvature(U, self.F)
            Mn = igl.massmatrix(U, self.F, igl.MASSMATRIX_TYPE_VORONOI)
            Minvn = sp.sparse.diags(1 / Mn.diagonal())
            Knn = Minvn.dot(K_Gn)
            
            # Dump time steps
            temp_file = "sim_" + str(i).zfill(6);
            saveFileVTU(timesteps_folder + "/" + temp_file, U, self.F,
                        verts_data = {"curvature": np.ascontiguousarray(Knn),
                                      "area_evolution": np.ascontiguousarray(area_p_v)})
            
            if renderstl:
                render_surfaces_gnuplot(fname = "_" + temp_file + ".stl",
                                        directory = timesteps_folder,
                                        V = U, F = self.F,
                                        extra_field = [])
            
            print("Saved {} / {} files.".format(i+1, self.iterations), end='\r')
        # Group dumped data
        saveGroupPVD(file_name = group_file,
                     data_path = timesteps_folder)
    
    def dump_final_parameterization(self, filename):
        # self.V_map_reg, self.F
        igl.write_triangle_mesh(filename, self.V_map_reg, self.F)
    
    
    @staticmethod
    def quantify_conformal_error():
        pass
    
    def __del__(self):
        pass

def smart_gradual_spheroid_shrink(coords, spheroid_dim, spheroid_type, theta_c = np.deg2rad(10), step_size = 0.0025):
    """
    Gradually shrink a target spheroid about the origin until 
    the capped points are all inscribed inside.
    """
    coords_ = copy.copy(coords)
    spheroid_dim_ = copy.copy(spheroid_dim)
    spheroid_type_ = copy.copy(spheroid_type)
    if spheroid_type_ == "prolate":
        spheroid_coord_func = cart2sph
        spheroid_func = prolate_objective_iso
        filter_coords = lambda thetas_, theta_c: (abs(thetas_) <= theta_c) | (abs(thetas_) >= (np.pi - theta_c))
    elif spheroid_type_ == "oblate":
        spheroid_coord_func = cart2sph
        spheroid_func = oblate_objective_iso
        filter_coords = lambda thetas_, theta_c: abs(np.pi/2 - thetas_) <= (theta_c)
        
    thetas_, phis_ = spheroid_coord_func(coords_[:, 0],
                                              coords_[:, 1],
                                              coords_[:, 2], spheroid_dim[2])
    f_coords_idx = filter_coords(thetas_, theta_c)
    f_coords = coords_[f_coords_idx, :]
    para_ = np.array([1.0, 0., 0., 0.])
    i = 1
    _, counted_pts = count_points_inside(para_, spheroid_dim_[0:2], f_coords, spheroid_type_)
    while counted_pts > 0:
        _, counted_pts = count_points_inside(para_, spheroid_dim_[0:2], f_coords, spheroid_type_)
        para_[0] -= step_size
        i += 1
    para_[0] -= step_size # Take out one additional step
    print("Number of steps:", i)
    return f_coords, para_[0]

def compute_bb(V):
    # Return bounding-box (8-points) coordinates
    bb_coords = np.zeros([8, 3])
    
    # Compute minimum and maximum coordinates
    min_coords = np.min(V, axis=0)
    max_coords = np.max(V, axis=0)
    
    # Compute coordinates
    bb_coords[0, 0] = min_coords[0]; bb_coords[0, 1] = min_coords[1]; bb_coords[0, 2] = min_coords[2]; # min, min, min
    bb_coords[1, 0] = max_coords[0]; bb_coords[1, 1] = min_coords[1]; bb_coords[1, 2] = min_coords[2]; # max, min, min
    bb_coords[2, 0] = max_coords[0]; bb_coords[2, 1] = max_coords[1]; bb_coords[2, 2] = min_coords[2]; # max, max, min
    bb_coords[3, 0] = min_coords[0]; bb_coords[3, 1] = max_coords[1]; bb_coords[3, 2] = min_coords[2]; # min, max, min
    bb_coords[4, 0] = min_coords[0]; bb_coords[4, 1] = min_coords[1]; bb_coords[4, 2] = max_coords[2]; # min, min, max
    bb_coords[5, 0] = max_coords[0]; bb_coords[5, 1] = min_coords[1]; bb_coords[5, 2] = max_coords[2]; # max, min, max
    bb_coords[6, 0] = max_coords[0]; bb_coords[6, 1] = max_coords[1]; bb_coords[6, 2] = max_coords[2]; # max, max, max
    bb_coords[7, 0] = min_coords[0]; bb_coords[7, 1] = max_coords[1]; bb_coords[7, 2] = max_coords[2]; # min, max, max
    return bb_coords

def computed_diag_bb(bb_coords):
    """
    Compute the diagonal distance of a given bounding box.
    """
    # Compute the Euclidean distance between two points
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    # Calculate the diagonal distances between all pairs of corners
    diagonal_distances = []
    for i in range(len(bb_coords)):
        for j in range(i + 1, len(bb_coords)):
            distance = euclidean_distance(bb_coords[i], bb_coords[j])
            diagonal_distances.append(distance)
    return max(diagonal_distances);

def is_triangles_orientation_consistent(vertices, triangles):
    """
    Check if the triangles orientation all pointing outward or all inward.
    This method is used to check if the radialy mapped surface is star-shaped.
    This is a low-computational cost method to check if a shape is not star-shaped.
    However, this approach works well for simple nonconvex contours.
    """
    # Calculate the centroid of the mesh
    centroid = np.mean(vertices, axis=0)
    # Iterate through each triangle in the mesh
    for triangle in triangles:
        # Get the vertices of the triangle
        v1, v2, v3 = [vertices[i] for i in triangle]

        # Calculate the normal vector of the triangle
        normal = np.cross(v2 - v1, v3 - v1)

        # Calculate the direction from the centroid to the normal vector
        direction = normal - centroid

        # Check if the direction vector points outward
        if np.dot(direction, normal) < 0:
            return False
    return True


if __name__ == "__main__":
    pass
