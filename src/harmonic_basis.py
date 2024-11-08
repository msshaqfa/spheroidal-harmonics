#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:10:29 2023

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
import numpy as np
import scipy, math, copy
from matplotlib import pyplot as plt
from matplotlib import cm, colors

from plot_helper import *
from spheroid_coordinates import *
from IO_helper import *
from render_stl import *

def basis_functions(max_n, theta, phi, spheroid_type):
    # This function can be optimized and further vectorized, but for readability I leave it like this :)
    Y_mat = np.zeros([theta.shape[0], (max_n+1)**2], dtype=complex)
    if spheroid_type == "prolate" or spheroid_type == "sphere":
        xi = np.cos(theta)
    elif spheroid_type == "oblate":
        xi = np.sin(theta)
    for n in range(max_n + 1):
        for m in range(-n, n+1):
            N_mn = np.sqrt( (2*n + 1) * math.factorial(abs(n - abs(m))) / (4*np.pi * math.factorial(n+abs(m))))
            if m >= 0:
                Y_mat[:, n**2 + n + m] =  N_mn * np.exp(-1j * m * phi) * scipy.special.lpmv(m, n, xi)
            else:
                Y_mat[:, n**2 + n + m] = (-1)**m * N_mn * (np.exp(-1j * abs(m) * phi) * scipy.special.lpmv(abs(m), n, xi)).conj()
    return Y_mat

def basis_reconstruction(coefs, rec_thetas, rec_phis, max_n, spheroid_type):
    rec_Y_mat = basis_functions(max_n, rec_thetas, rec_phis, spheroid_type)
    rec_rs = np.zeros(rec_thetas.shape, dtype=complex)
    for n in range(0, max_n+1):
        for m in range(-n, n+1):
            rec_rs += coefs[n**2 + n + m] * rec_Y_mat[:, n**2 + n + m]    
    return rec_rs

def basis_reconstruction2(coefs, rec_thetas, rec_phis, max_n, spheroid_type):
    rec_Y_mat = basis_functions(max_n, rec_thetas, rec_phis, spheroid_type)
    rec_x = np.zeros(rec_thetas.shape, dtype=complex)
    rec_y = np.zeros(rec_thetas.shape, dtype=complex)
    rec_z = np.zeros(rec_thetas.shape, dtype=complex)
    for n in range(0, max_n+1):
        for m in range(-n, n+1):
            rec_x += coefs[n**2 + n + m, 0] * rec_Y_mat[:, n**2 + n + m]
            rec_y += coefs[n**2 + n + m, 1] * rec_Y_mat[:, n**2 + n + m]
            rec_z += coefs[n**2 + n + m, 2] * rec_Y_mat[:, n**2 + n + m]
    return rec_x, rec_y, rec_z

def recursive_basis_reconstruction(coefs_x, coefs_y, coefs_z,
                                   rec_thetas, rec_phis, F, max_n,
                                   spheroid_type, filenames_dict,
                                   increment = 1, renderstl = False):
    # Dump files names
    timesteps_folder = filenames_dict["timesteps_folder"]
    timesteps_files = filenames_dict["timesteps_files"]
    group_folder = filenames_dict["group_folder"]
    group_file = filenames_dict["group_file"]
    createIOFolder(timesteps_folder)
    
    # Compute basis
    rec_Y_mat = basis_functions(max_n, rec_thetas, rec_phis, spheroid_type)
    
    # Pre-initialize
    rec_x = np.zeros(rec_thetas.shape, dtype="float32")
    rec_y = copy.copy(rec_x)
    rec_z = copy.copy(rec_x)
    temp = 0
    
    for n in range(0, max_n+1):
        for m in range(-n, n+1):
            rec_x += (coefs_x[n**2 + n + m] * rec_Y_mat[:, n**2 + n + m]).real
            rec_y += (coefs_y[n**2 + n + m] * rec_Y_mat[:, n**2 + n + m]).real
            rec_z += (coefs_z[n**2 + n + m] * rec_Y_mat[:, n**2 + n + m]).real
        temp += 1
        # Mesh IO
        if temp == increment:
            temp_file = "rec_deg_" + str(n).zfill(3)
            temp_rec_V = np.array([rec_x, rec_y, rec_z]).T
            temp_rad = np.sqrt(rec_x**2 + rec_y**2 + rec_z**2)
            saveFileVTU(timesteps_files + "/" + temp_file,
                        temp_rec_V, F,
                        verts_data = {"radial_distance": np.ascontiguousarray(temp_rad)}
                        )
            if renderstl:
                render_surfaces_gnuplot(fname = "_" + temp_file + ".stl",
                                        directory = timesteps_folder,
                                        V = temp_rec_V, F = F,
                                        extra_field = [])
            
            print("Saved rec deg {} / {} files.".format(n+1, max_n+1), end='\r')
            temp = 0
    # Group dumped data
    saveGroupPVD(file_name = group_file,
                 data_path = timesteps_folder)
    return np.array([rec_x, rec_y, rec_z]).T


def math_basis_functions(max_n, theta, phi, spheroid_type):
    if spheroid_type == "prolate" or spheroid_type == "sphere":
        xi = np.cos(theta)
    elif spheroid_type == "oblate":
        xi = np.sin(theta)
    Y_mat = np.zeros([theta.shape[0], theta.shape[1], (max_n+1)**2], dtype=complex)
    for n in range(max_n + 1):
        for m in range(-n, n+1):
            N_mn = np.sqrt( (2*n + 1) * math.factorial(abs(n - abs(m))) / (4*np.pi * math.factorial(n+abs(m))))
            if m >= 0:
                Y_mat[:, :, n**2 + n + m] = N_mn * np.exp(-1j * m * phi) * scipy.special.lpmv(m, n, xi)
            else:
                Y_mat[:, :, n**2 + n + m] = (-1)**abs(m) * ( N_mn * np.exp(-1j * abs(m) * phi) * scipy.special.lpmv(abs(m), n, xi) ).conj()
    return Y_mat

def math_basis_reconstruction(coefs, rec_thetas, rec_phis, max_n, spheroid_type):
    rec_Y_mat = math_basis_functions(max_n, rec_thetas, rec_phis, spheroid_type)
    rec_rs = np.zeros(rec_thetas.shape, dtype=complex)
    for n in range(0, max_n+1):
        for m in range(-n, n+1):
            rec_rs += coefs[n**2 + n + m] * rec_Y_mat[:, :, n**2 + n + m]    
    return rec_rs.real

def plot_math_rec(rec_max_n, coefsx, coefsy, coefsz, spheroid_type, subdivgrid):
    for max_n in rec_max_n:
        if spheroid_type == "prolate" or spheroid_type == "sphere":
            rec_thetas, rec_phis = np.linspace(0, np.pi, subdivgrid), np.linspace(0, 2*np.pi, subdivgrid)
        elif spheroid_type == "oblate":
            rec_thetas, rec_phis = np.linspace(-np.pi/2, np.pi/2, subdivgrid), np.linspace(0, 2*np.pi, subdivgrid)
        grid_thetas, grid_phis = np.meshgrid(rec_thetas, rec_phis)
        rec_xs = math_basis_reconstruction(coefsx, grid_thetas, grid_phis, max_n, spheroid_type)
        rec_ys = math_basis_reconstruction(coefsy, grid_thetas, grid_phis, max_n, spheroid_type)
        rec_zs = math_basis_reconstruction(coefsz, grid_thetas, grid_phis, max_n, spheroid_type)
        rec_rs = np.sqrt(rec_xs**2 + rec_ys**2 + rec_zs**2)
        fmax, fmin = rec_rs.max(), rec_rs.min()
        fcolors = (rec_rs - fmin)/(fmax - fmin)
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(rec_xs, rec_ys, rec_zs, facecolors=cm.seismic(fcolors))
        plt.title('Surface reconstruction at n = {}'.format(str(max_n)))
        plt.axis('off')
        plt.show()

def plot_sph_mn(n, m, f, mu, spheroid_type, subgrid=200):
    phi = np.linspace(0, 2*np.pi - 0.009, subgrid)
    if spheroid_type == "prolate":
        theta = np.linspace(0, np.pi, subgrid)
        phi, theta = np.meshgrid(phi, theta)
        xi = np.cos(theta)
        spheroid2catr = prolate2cart
    elif spheroid_type == "oblate":
        theta = np.linspace(-np.pi/2, np.pi/2, subgrid)
        phi, theta = np.meshgrid(phi, theta)
        xi = np.sin(theta)
        spheroid2catr = oblate2cart
    elif spheroid_type == "sphere":
        theta = np.linspace(0, np.pi, subgrid)
        phi, theta = np.meshgrid(phi, theta)
        xi = np.cos(theta)
        spheroid2catr = sph2cart
    if spheroid_type == "sphere":
        x, y, z = spheroid2catr(theta, phi, mu)
    else:
        x, y, z = spheroid2catr(theta, phi, f, mu)
        
    
    N_mn = np.sqrt( (2*n + 1) * math.factorial(abs(n - abs(m))) / (4*np.pi * math.factorial(n+abs(m))))
    r = N_mn * ( np.exp(-1j * m * phi) * scipy.special.lpmv(abs(m), n, xi) )
    if m < 0: r = (-1)**abs(m) * r.conj()
    r = r.real()
    scamap = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    fcolors = scamap.to_rgba(r)
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, facecolors = fcolors, cmap=cm.coolwarm,
                       linewidth=0., antialiased=False, edgecolor="black")
    ax.set_axis_off()
    set_axes_equal(ax)
    plt.title('Real plot for the spheroid basis m = {}, n = {}'.format(str(m), str(n)))
    plt.show()
    

def shape_descriptors(coefs, max_n):
    Dr = np.zeros((max_n,))
    # Rotation-invariant
    for n in range(max_n):
        for m in range(-n, n+1):
            Dr[n] += coefs[n**2 + n + m].real**2 + coefs[n**2 + n + m].imag**2
    # Scale-invariant
    Dr /= Dr[1]
    return Dr

def shape_descriptors2(coefs, max_n):
    Dr_x = np.zeros((max_n,))
    Dr_y = np.zeros((max_n,))
    Dr_z = np.zeros((max_n,))
    # Rotation-invariant
    for n in range(max_n):
        for m in range(-n, n+1):
            Dr_x[n] += coefs[n**2 + n + m, 0].real**2 + coefs[n**2 + n + m, 0].imag**2
            Dr_y[n] += coefs[n**2 + n + m, 1].real**2 + coefs[n**2 + n + m, 1].imag**2
            Dr_z[n] += coefs[n**2 + n + m, 2].real**2 + coefs[n**2 + n + m, 2].imag**2
    # Scale-invariant
    Dr_x /= Dr_x[1]
    Dr_y /= Dr_y[1]
    Dr_z /= Dr_z[1]
    return Dr_x, Dr_y, Dr_z


if __name__ == "__main__":    
    plot_sph_mn(6, 2, 5, 0.5, "oblate", 50)
    plot_sph_mn(6, 2, 5, 0.5, "prolate", 50)
    plot_sph_mn(6, 2, 0, 1.0, "sphere", 50)
