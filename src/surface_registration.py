#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:54:08 2023

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


def register_surface(coords, verts_limit = 20000):
    """
    Register any surface by rotating them into a canonical coordinates. This includes:
        1- transforming the centroid into the origin
        2- rotate the surface
    """
    
    # First step translate your data to the origin (mean)
    centroid = np.mean(coords, axis=0)
    
    # Sample a subset of the object, we don't need all data to define general orientation
    if len(coords[:,1]) > verts_limit:
        sampled_verts = int(verts_limit)
        _ip = np.random.permutation(len(coords[:,1]))
        _idel = _ip[:sampled_verts]
        subcoords =  coords[_idel, :]
    else:
        subcoords = coords
    
    # Second step apply SVD for the surface data
    U, Sigma, Vt = np.linalg.svd(subcoords.T)
    
    return centroid, U


if __name__ == "__main__":
    pass