#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:43:09 2023

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

# Spherical coordinates
def sph2cart(theta, phi, r=1, r2 = 1):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cart2sph(x, y, z, r=1, r2 = 1):
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    theta = np.arctan2(z, (np.sqrt(x**2 + y**2))); theta = np.pi/2 - theta
    return theta, phi


# Oblate coordinates
def cart2oblate(x, y, z, f, zeta):
    """Linearized mapping"""
    xx = x / (f * np.cosh(zeta))
    yy = y / (f * np.cosh(zeta))
    zz = z / (f * np.sinh(zeta))    
    
    theta = np.arctan2(zz, (np.sqrt(xx**2 + yy**2))); # Radial linearized assumption
        
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi

def cart2oblate2(x, y, z, f, zeta):
    """Nonlinear analytic mapping"""
    theta = ( ( np.arccosh( (np.sqrt(x**2 + y**2) + 1j * z) / f  ) )).imag # Analytic with hyperbolic normals
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi


def oblate2cart(theta, phi, f, zeta):
    x = f * np.cosh(zeta) * np.cos(theta) * np.cos(phi)
    y = f * np.cosh(zeta) * np.cos(theta) * np.sin(phi)
    z = f * np.sinh(zeta) * np.sin(theta)
    return x, y, z

# Prolate coordinates
def cart2prolate(x, y, z, f, zeta):
    """Linearized mapping"""
    xx = x / (f * np.sinh(zeta))
    yy = y / (f * np.sinh(zeta))
    zz = z / (f * np.cosh(zeta))
    
    theta = np.arctan2(zz, (np.sqrt(xx**2 + yy**2))); theta = np.pi/2 - theta # Radial linearized assumption
    
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi

def cart2prolate2(x, y, z, f, zeta):
    """Nonlinear analytic mapping"""
    theta = ( ( np.arccosh( ( 1j * np.sqrt(x**2 + y**2) + z ) / f  ) )).imag # Analytic with hyperbolic normals
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi

def cart2oblate3(x, y, z, f):
    """Nonlinear analytic mapping"""
    theta = ( ( np.arccosh( (np.sqrt(x**2 + y**2) + 1j * z) / f  ) )).imag # Analytic with hyperbolic normals
    zeta = ( ( np.arccosh( (np.sqrt(x**2 + y**2) + 1j * z) / f  ) )).real # Analytic with hyperbolic radius
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi, zeta

def cart2prolate3(x, y, z, f):
    """Nonlinear analytic mapping"""
    theta = ( ( np.arccosh( ( 1j * np.sqrt(x**2 + y**2) + z ) / f  ) )).imag # Analytic with hyperbolic normals
    zeta = ( ( np.arccosh( ( 1j * np.sqrt(x**2 + y**2) + z ) / f  ) )).real # Analytic with hyperbolic radius
    phi = np.arctan2(y, x); neg_id = np.where(phi < 0); phi[neg_id] += 2*np.pi
    return theta, phi, zeta

def prolate2cart(theta, phi, f, zeta):
    x = f * np.sinh(zeta) * np.sin(theta) * np.cos(phi)
    y = f * np.sinh(zeta) * np.sin(theta) * np.sin(phi)
    z = f * np.cosh(zeta) * np.cos(theta)
    return x, y, z
