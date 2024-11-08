#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 23:39:40 2023

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

import time, copy, os, sys, scipy, warnings
from pprint import pprint
from stl import mesh
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import special
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator
from scipy.special import *

# Internal codes
from plot_helper import *

def canonical_coordinates(coords):
    '''
    Shift the centroid of the surface and remove rotations
    '''
    # First step translate your data to the origin (mean)
    centroid = np.mean(coords, axis=0) 
    coords -= centroid
    
    # Apply SVD for the data
    U, Sigma, Vt = np.linalg.svd(coords.T)
    coords_unrotated = U.T @ coords.T
    
    return centroid, U, coords_unrotated


def fit_spheroids(coords, nonlinear_proj=False, printed=False):
    '''
    Fit spheroids with least squares algorithm. This function first fits an 
    ellipsoid then decides whether it is an oblate or prolate object.
    After that, we try to fit an oblate or prolate objects to the given dataset.
    
    Input:
        Coords (numpy array): #V x 3
        
    Output:
        C_sph: array that returns the spheroid fit parameters
        spheroid: text of spheroid type "oblate" of "prolate"
    '''
    coords = coords.T
    x_, y_, z_ = coords[0, :], coords[1, :], coords[2, :]
    
    A = np.zeros([x_.shape[0], 3])
    f = np.ones([x_.shape[0], 1])
    A[:, 0] = x_ * x_
    A[:, 1] = y_ * y_
    A[:, 2] = z_ * z_

    C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)
    C_ellip = np.array([np.sqrt(1/C[0]), np.sqrt(1/C[1]), np.sqrt(1/C[2])])
    if printed: print("Stage 1 - Ellipsoidal parameters are: a = {}, b = {}, c = {}".format(C_ellip[0],
                                                                                C_ellip[1],
                                                                                C_ellip[2]))
    
    # Classify the spheroid type
    dist = np.zeros([3, ])
    dist[0] = abs(C_ellip[0] - C_ellip[1])
    dist[1] = abs(C_ellip[1] - C_ellip[2])
    dist[2] = abs(C_ellip[0] - C_ellip[2])
    
    min_dist = np.where(dist == dist.min())[0][0]
    if min_dist == 0:
        aa = np.mean([C_ellip[0], C_ellip[1]])
        cc = C_ellip[2]
    elif min_dist == 1:
        aa = np.mean([C_ellip[1], C_ellip[2]])
        cc = C_ellip[0]
    elif min_dist == 2:
        aa = np.mean([C_ellip[0], C_ellip[2]])
        cc = C_ellip[1]
    
    if aa <= cc:
        spheroid = "prolate"
    else:
        spheroid = "oblate"
    if printed: print("The fitted object is: {}".format(spheroid))

    # Refit an optimized spheroid
    A = np.zeros([x_.shape[0], 2])
    if spheroid == "prolate":
        A[:, 0] = x_ * x_
        A[:, 1] = y_ * y_ + z_ * z_
    elif spheroid == "oblate":
        A[:, 0] = x_ * x_ + y_ * y_
        A[:, 1] = z_ * z_

    C, residules, rank, singval = np.linalg.lstsq(A,f, rcond=None)
    C_sph = np.zeros((4,1))
    axis_ = np.array([np.sqrt(1/C[0]), np.sqrt(1/C[1])])
    if spheroid == "prolate":
        C_sph[0] = axis_.min()
        C_sph[1] = axis_.max()
        C_sph[2] = np.sqrt(abs(C_sph[1]**2 - C_sph[0]**2))
        #C_sph[3] = np.arcsinh(abs(aa) / C_sph[2])
        C_sph[3] = np.arcsinh(abs(C_sph[0]) / C_sph[2])
        if printed: print("Stage 2 - Spheroid parameters of the {}: a = {}, c = {}, f = {}, zeta = {}".format(
            spheroid, C_sph[1], C_sph[0], C_sph[2], C_sph[3]))
    elif spheroid == "oblate":
        C_sph[0] = axis_.max()
        C_sph[1] = axis_.min()
        C_sph[2] = np.sqrt(abs(C_sph[1]**2 - C_sph[0]**2))
        #C_sph[3] = np.arccosh(abs(aa) / C_sph[2])
        C_sph[3] = np.arccosh(abs(C_sph[0]) / C_sph[2])
        if printed: print("Stage 2 - Spheroid parameters of the {}: a = {}, c = {}, f = {}, zeta = {}".format(
            spheroid, C_sph[0], C_sph[1], C_sph[2], C_sph[3]))
    r2 = 1 - residules / (x_.shape[0] * x_.var())
    if printed: print("Fittness quality R^2 = {}".format(r2))
    return C_sph, spheroid

# Additional functions for optimization-based spheroidal fit (inscribed spheroid objective)
def ellipsoid_objective(params, coords):
    # Objective function for fitting an ellipsoid (optimization-based)
    x, y, z = coords
    c1, c2, c3 = params
    return np.min( c1 * (x)**2 + c2 * (y)**2 + c3 * (z)**2 )

def prolate_objective(params, coords):
    # Objective function for fitting a prolate (optimization-based)
    x, y, z = coords
    c1, c2 = params
    return np.min( c1 * (x)**2 + c2 * (y**2 + z**2) )

def oblate_objective(params, coords):
    # Objective function for fitting an oblate (optimization-based)
    x, y, z = coords
    c1, c2 = params
    return np.min( c1 * (x**2 + y**2) + c2 * (z)**2 )

def constraint_ellipsoid(params, coords):
    # Objective function for fitting an ellipsoid (optimization-based)
    x, y, z = coords
    c1, c2, c3 = params
    return np.min( c1 * (x)**2 + c2 * (y)**2 + c3 * (z)**2 ) - 1.

def constraint_prolate(params, coords):
    # Objective function for fitting an ellipsoid (optimization-based)
    x, y, z = coords
    c1, c2 = params
    return np.min( c1 * (x)**2 + c2 * (y**2 + z**2) ) - 1.

def constraint_oblate(params, coords):
    # Objective function for fitting an ellipsoid (optimization-based)
    x, y, z = coords
    c1, c2 = params
    return np.min( c1 * (x**2 + y**2) + c2 * (z)**2 ) - 1.

def prolate_objective_iso(params, spheroid_axes, coords):
    # Objective function for fitting a prolate (isotropic optimization-based)
    x, y, z = copy.copy(coords)
    a, c = copy.copy(spheroid_axes)
    multiplier, x0, y0, z0 = copy.copy(params)
    return (1 / (multiplier * c) )**2 * (z - z0)**2 + (1/ (multiplier * a) )**2 * ( (y - y0)**2 + (x - x0)**2 )

def oblate_objective_iso(params, spheroid_axes, coords):
    # Objective function for fitting an oblate (isotropic optimization-based)
    x, y, z = copy.copy(coords)
    a, c = copy.copy(spheroid_axes)
    multiplier, x0, y0, z0 = copy.copy(params)
    return (1/ (multiplier * a) )**2 * ( (x - x0)**2 + (y - y0)**2) + (1/ (multiplier * c) )**2 * (z - z0)**2

def prolate_objective_opt(params, coords):
    # Objective function for fitting a prolate (optimization-based)
    x, y, z = coords
    a, c, x0, y0, z0 = params
    return (1/a)**2 * (z - z0)**2 + (1/c)**2 * ((y - y0)**2 + (x - x0)**2)

def oblate_objective_opt(params, coords):
    # Objective function for fitting an oblate (optimization-based)
    x, y, z = coords
    a, c, x0, y0, z0 = params
    return (1/a)**2 * ((x - x0)**2 + (y - y0)**2) + (1/c)**2 * (z - z0)**2

def count_points_inside(params, spheroid_axes, coords, spheroid_type):
    # Count the percentage of points inside the spheroid    
    coords_ = tuple(copy.copy(coords).T)
    num_ = len(coords)
    params_ = copy.copy(params)
    spheroid_axes_ = copy.copy(spheroid_axes)
    multiplier, x0, y0, z0 = params_
    if spheroid_type == "prolate":
        spheroid_func = prolate_objective_iso
    elif spheroid_type == "oblate":
        spheroid_func = oblate_objective_iso
    spheroid_proj = spheroid_func(params_, spheroid_axes_, coords_)
    return np.sum(spheroid_proj <= 1.0) / num_, np.sum(spheroid_proj <= 1.0)

def count_points_inside_opt(params, spheroid_axes, coords, spheroid_type):
    # Count the percentage of points inside the spheroid (Stage 2)    
    coords_ = tuple(copy.copy(coords).T)
    num_ = len(coords)
    params_ = copy.copy(params)
    spheroid_axes_ = copy.copy(spheroid_axes)
    # a, c, x0, y0, z0 = params_
    if spheroid_type == "prolate":
        spheroid_func = prolate_objective_opt
    elif spheroid_type == "oblate":
        spheroid_func = oblate_objective_opt
    spheroid_proj = spheroid_func(params_, coords_)
    return np.sum(spheroid_proj <= 1.0) / num_, np.sum(spheroid_proj <= 1.0)

def gradual_spheroid_shrink(coords, spheroid_axes, spheroid_type, step_size = 0.001):
    """
    Gradual shrink about the origin of 
    the spheroid until no points are inside the spheroid
    """
    coords_ = copy.copy(coords)
    spheroid_axes_ = copy.copy(spheroid_axes)
    spheroid_type_ = copy.copy(spheroid_type)
    para_ = np.array([1.0, 0., 0., 0.])
    i = 1
    _, counted_pts = count_points_inside(para_, spheroid_axes_, coords_, spheroid_type_)
    while counted_pts > 0:
        _, counted_pts = count_points_inside(para_, spheroid_axes_, coords_, spheroid_type_)
        # print(i, counted_pts)
        para_[0] -= step_size
        i += 1
        # print(spheroid_type_, para_[0], spheroid_axes_)
    para_[0] -= step_size # Take out one additional step
    print("Number of steps:", i)
    
    return para_[0]


def fit_inscribed_spheroids(coords, printed=False):
    '''
    Fits (optimization-based) the largest inscribed spheroids within a stone via minimization algorithms.
    This function first fits an ellipsoid then decides whether it is an oblate or prolate object.
    After that, we try to fit an oblate or prolate objects to the given dataset (all fits are inside the stones).
    Input:
        coords (numpy array): #V x 3
        
    Output:
        C_sph: array that returns the inscribed spheroid fit parameters
        spheroid: text of spheroid type "oblate" of "prolate"
    '''
    coords = tuple(coords.T)
    
    # Inequlaity Constraint definition
    ineqconst_ellipsoid = lambda params: constraint_ellipsoid(params, coords)
    ineqconst_prolate = lambda params: constraint_prolate(params, coords)
    ineqconst_oblate = lambda params: constraint_oblate(params, coords)
    
    constraints_ellipsoid = {'type': 'ineq', 'fun': ineqconst_ellipsoid}
    constraints_prolate = {'type': 'ineq', 'fun': ineqconst_prolate}
    constraints_oblate = {'type': 'ineq', 'fun': ineqconst_oblate}
    
    # 1st stage
    opt_res = minimize(ellipsoid_objective, np.ones(3),
                       args=(coords,), constraints=constraints_ellipsoid)
    if not opt_res.success: warnings.warn("The solver did not converge for stage 1.")
    C = opt_res.x
    C_ellip = np.array([np.sqrt(1/C[0]), np.sqrt(1/C[1]), np.sqrt(1/C[2])])
    
    if printed:
        print("Solver stage 1: ", opt_res.message)
        print("Stage 1 - The inscribed ellipsoidal parameters are: a = {}, b = {}, c = {}".format(C_ellip[0],
                                                                                C_ellip[1],
                                                                                C_ellip[2]))
    
    # Classify the spheroid type
    dist = np.zeros([3, ])
    dist[0] = abs(C_ellip[0] - C_ellip[1])
    dist[1] = abs(C_ellip[1] - C_ellip[2])
    dist[2] = abs(C_ellip[0] - C_ellip[2])
    
    min_dist = np.where(dist == dist.min())[0][0]
    if min_dist == 0:
        aa = np.mean([C_ellip[0], C_ellip[1]])
        cc = C_ellip[2]
    elif min_dist == 1:
        aa = np.mean([C_ellip[1], C_ellip[2]])
        cc = C_ellip[0]
    elif min_dist == 2:
        aa = np.mean([C_ellip[0], C_ellip[2]])
        cc = C_ellip[1]
    
    if aa <= cc:
        spheroid = "prolate"
    else:
        spheroid = "oblate"
    if printed: print("The fitted object is: {}".format(spheroid))
    
    # 2nd stage
    if spheroid == "prolate":
        obj_spheroid = prolate_objective
        ineqconst_spheroid = constraints_prolate
    elif spheroid == "oblate":
        obj_spheroid = oblate_objective
        ineqconst_spheroid = constraints_oblate
    
    initial_axes_spheroid = np.array([(1/aa)**2, (1/cc)**2])
    
    opt_res_spheroid = minimize(obj_spheroid, initial_axes_spheroid,
                                args=(coords,), constraints=ineqconst_spheroid)
    if not opt_res_spheroid.success: warnings.warn("The solver did not converge for stage 2.")
    if printed: print("Solver stage 2: ", opt_res_spheroid.message)
    C2 = opt_res_spheroid.x
    C_sph = np.zeros((4,1))
    axis_ = np.array([np.sqrt(1/C2[0]), np.sqrt(1/C2[1])])
    
    if spheroid == "prolate":
        C_sph[0] = axis_.min()
        C_sph[1] = axis_.max()
        C_sph[2] = np.sqrt(abs(C_sph[1]**2 - C_sph[0]**2))
        C_sph[3] = np.arcsinh(abs(C_sph[0]) / C_sph[2])
        if printed: print("Stage 2 -Inscrbied spheroid parameters of the {}: a = {}, c = {}, f = {}, zeta = {}".format(
            spheroid, C_sph[1], C_sph[0], C_sph[2], C_sph[3]))
    elif spheroid == "oblate":
        C_sph[0] = axis_.max()
        C_sph[1] = axis_.min()
        C_sph[2] = np.sqrt(abs(C_sph[1]**2 - C_sph[0]**2))
        C_sph[3] = np.arccosh(abs(C_sph[0]) / C_sph[2])
        if printed: print("Stage 2 - Inscrbied spheroid parameters of the {}: a = {}, c = {}, f = {}, zeta = {}".format(
            spheroid, C_sph[0], C_sph[1], C_sph[2], C_sph[3]))
    
    return C_sph, spheroid




# Generate tests of spheroids
def spheroid_func(x, y, z, a, c):
    return (x**2 + y**2)/a**2 + z**2/c**2 - 1


def prolate_like_noise(f = 5, zeta = 0.3, init_sample = 1000, fin_sample = 200,
                       rot_x = 0., rot_y = 0., rot_z = 0., noise_ampl = 0.1, figure=True):
    theta = np.linspace(0, np.pi - 0.009, num=init_sample)
    phi = np.linspace(0, 2*np.pi - 0.009, num=init_sample)
    theta, phi = np.meshgrid(theta, phi)
    
    x = (f * np.sinh(zeta) * np.sin(theta) * np.cos(phi)).ravel()
    y = (f * np.sinh(zeta) * np.sin(theta) * np.sin(phi)).ravel()
    z = (f * np.cosh(zeta) * np.cos(theta)).ravel()
    
    # Add noise (Gaussian)
    noise_x = np.random.normal(0, 1, len(x))
    noise_y = np.random.normal(0, 1, len(x))
    noise_z = np.random.normal(0, 1, len(x))
    x += noise_ampl * noise_x
    y += noise_ampl * noise_y
    z += noise_ampl * noise_z
    
    # Random sampling
    sampled_rows = int(fin_sample)
    _ip = np.random.permutation(len(x))
    _idel = _ip[:sampled_rows]
    x = x[_idel]
    y = y[_idel]
    z = z[_idel]
    
    # Rotate data
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot_x), -np.sin(rot_x)],
                   [0, np.sin(rot_x), np.cos(rot_x)]]
                 )
    Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                   [0, 1, 0],
                   [-np.sin(rot_y), 0, np.cos(rot_y)]]
                 )
    Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0],
                   [np.sin(rot_z), np.cos(rot_z), 0],
                  [0, 0, 1]]
                 )
    rotated_verts = Rx @ Ry @ Rz @ np.array([x, y, z])    
    x = rotated_verts[0, :]
    y = rotated_verts[1, :]
    z = rotated_verts[2, :]
    
    if figure:
        fig = plt.figure()
        colmap = cm.ScalarMappable(cmap=cm.hsv)
        r = np.sqrt(x**2 + y**2 + z**2)
        colmap.set_array(r)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=cm.hsv(r/max(r)), marker='o', s = 0.5)
        set_axes_equal(ax)
        plt.title("Random Prolate object")
        plt.show()
    print("Prolate object generated with f = {}, zeta = {}.".format(f, zeta))
    return x, y, z

def oblate_like_noise(f = 5, zeta = 0.3, init_sample = 1000, fin_sample = 200,
                      rot_x = 0., rot_y = 0., rot_z = 0., noise_ampl = 0.1, figure=True):
    theta = np.linspace(0, np.pi - 0.009, num=init_sample)
    phi = np.linspace(0, 2*np.pi - 0.009, num=init_sample)
    theta, phi = np.meshgrid(theta, phi)
    
    x = (f * np.cosh(zeta) * np.sin(theta) * np.cos(phi)).ravel()
    y = (f * np.cosh(zeta) * np.sin(theta) * np.sin(phi)).ravel()
    z = (f * np.sinh(zeta) * np.cos(theta)).ravel()
    
    # Add noise (Gaussian)
    noise_x = np.random.normal(0, 1, len(x))
    noise_y = np.random.normal(0, 1, len(x))
    noise_z = np.random.normal(0, 1, len(x))
    x += noise_ampl * noise_x
    y += noise_ampl * noise_y
    z += noise_ampl * noise_z
    
    # Random sampling
    sampled_rows = int(fin_sample)
    _ip = np.random.permutation(len(x))
    _idel = _ip[:sampled_rows]
    x = x[_idel]
    y = y[_idel]
    z = z[_idel]
    
    # Rotate data
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot_x), -np.sin(rot_x)],
                   [0, np.sin(rot_x), np.cos(rot_x)]]
                 )
    Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                   [0, 1, 0],
                   [-np.sin(rot_y), 0, np.cos(rot_y)]]
                 )
    Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0],
                   [np.sin(rot_z), np.cos(rot_z), 0],
                  [0, 0, 1]]
                 )
    rotated_verts = Rx @ Ry @ Rz @ np.array([x, y, z])
    x = rotated_verts[0, :]
    y = rotated_verts[1, :]
    z = rotated_verts[2, :]
    
    if figure:
        fig = plt.figure()
        colmap = cm.ScalarMappable(cmap=cm.hsv)
        r = np.sqrt(x**2 + y**2 + z**2)
        colmap.set_array(r)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=cm.hsv(r/max(r)), marker='o', s = 0.7)
        set_axes_equal(ax)
        plt.title("Random Oblate object")
        plt.show()
    print("Oblate object generated with f = {}, zeta = {}.".format(f, zeta))
    return x, y, z

def plot_parametric_prolate(f = 10, zeta = 0.5, subdivgrid = 40):    
    theta = np.linspace(0, np.pi, num=subdivgrid)
    phi = np.linspace(0, 2*np.pi, num=subdivgrid)
    theta, phi = np.meshgrid(theta, phi)
    
    x = f * np.sinh(zeta) * np.sin(theta) * np.cos(phi)
    y = f * np.sinh(zeta) * np.sin(theta) * np.sin(phi)
    z = f * np.cosh(zeta) * np.cos(theta)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.gray,
                           linewidth=0.1, antialiased=False, edgecolors='k')
    plt.title("Parametric Prolate Spheroid")
    set_axes_equal(ax)
    plt.show()

def plot_parametric_oblate(f = 10, zeta = 0.5, subdivgrid = 40):
    theta = np.linspace(-np.pi/2, np.pi/2, num=subdivgrid)
    phi = np.linspace(0, 2*np.pi - 0.009, num=subdivgrid)
    theta, phi = np.meshgrid(theta, phi)

    x = f * np.cosh(zeta) * np.cos(theta) * np.cos(phi)
    y = f * np.cosh(zeta) * np.cos(theta) * np.sin(phi)
    z = f * np.sinh(zeta) * np.sin(theta)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.gray,
                           linewidth=0.1, antialiased=False, edgecolors='k')
    plt.title("Parametric Oblate Spheroid")
    set_axes_equal(ax)
    plt.show()


# Spheroidal area functions
eli_prolate = lambda a_i, c_i: max([np.sqrt( abs(1 - (a_i/c_i)**2) ), 0.99]) 
eli_oblate = lambda a_i, c_i: max([np.sqrt( abs(1 - (c_i/a_i)**2) ), 0.99])
A_prolate = lambda a_i, c_i, eli: 2 * np.pi * a_i**2 + 2 * np.pi * ( a_i * c_i / eli) * np.arcsin(eli)
A_oblate = lambda a_i, c_i, eli: 2 * np.pi * a_i**2 + np.pi *  (c_i**2 / eli) * np.log((1+eli)/(1-eli))


if __name__ == "__main__":
    # Test fitting on random noise
    
    f_oblate = 3
    zeta_oblate = 0.3
    
    f_prolate = 7
    zeta_prolate = 0.3
    
    noise_ampl = 0.2
    
    x_ob, y_ob, z_ob = oblate_like_noise(f = f_oblate, zeta = zeta_oblate,
                                         init_sample = 3000, fin_sample = 2000,
                                         noise_ampl = noise_ampl, rot_x = np.pi/2)
    print("The unrotated version")
    x_ob_, y_ob_, z_ob_ = oblate_like_noise(f = f_oblate, zeta = zeta_oblate,
                                            init_sample = 3000, fin_sample = 2000,
                                            noise_ampl = noise_ampl)
    
    x_pr, y_pr, z_pr = prolate_like_noise(f = f_prolate, zeta = zeta_prolate,
                                          init_sample = 3000, fin_sample = 2000,
                                          noise_ampl = noise_ampl, rot_x = np.pi/6)
    print("The unrotated version")
    x_pr_, y_pr_, z_pr_ = prolate_like_noise(f = f_prolate, zeta = zeta_prolate,
                                             init_sample = 3000, fin_sample = 2000,
                                             noise_ampl = noise_ampl, rot_y = np.pi/2)
    
    # Test fitting on real surfaces
    filename = 'Stone.stl'
    load_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/input_geometries/"
    loaded_object = STLAnalyzer(load_path + filename)
    coords = loaded_object.vertices
    print("Data point size: ", coords.shape)
    
    # Normalize surfaces (transformation and location)
    centroid, U, coords_unrotated = canonical_coordinates(coords)
    r_map = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
    r_map2 = np.sqrt(coords_unrotated[0, :]**2 + coords_unrotated[1, :]**2 
                     + coords_unrotated[2, :]**2)

    plot_3D_pc(coords[:, 0], coords[:, 1], coords[:, 2], r_map)
    plot_3D_pc(coords_unrotated[0, :], coords_unrotated[1, :],
               coords_unrotated[2, :], r_map2)
    
    # Least-squares to fit a spheroid on the standard registration
    C_sph, spheroid_type = fit_spheroids(coords_unrotated.T, True)
    
    if spheroid_type == "prolate":
        para_draw_func = plot_parametric_prolate
        th_y = np.pi/2
        R_y = np.array([[np.cos(th_y), 0, np.sin(th_y)],
                        [0, 1, 0], [-np.sin(th_y), 0, np.cos(th_y)]])
        coords_unrotated = R_y @ coords_unrotated
    else:
        para_draw_func = plot_parametric_oblate
    
    x_ = coords_unrotated[0, :]
    y_ = coords_unrotated[1, :]
    z_ = coords_unrotated[2, :]
    
    plot_implicit_with_scatter(spheroid_func, x_, y_, z_, C_sph[0], C_sph[1],
                               bbox=(-r_map2.max(), r_map2.max()))



    






