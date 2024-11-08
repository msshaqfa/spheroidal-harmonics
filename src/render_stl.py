"""
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

import random
import numpy as np

def export_DAT_files_gnuplot(fname, directory, V, F, extra_field = []):
    if extra_field == []:
        pass
    else:
        print("extra_field", extra_field.shape)
        print("V", V.shape)
        V  = np.concatenate((V, extra_field[:, 0]), axis=1)
    # Write *.dat file
    rearranged_vs = []
    for ff in F:
        for vv in ff:
            temp = V[vv].tolist()
            if extra_field != []:
                temp.append(extra_field[vv, 0])
            rearranged_vs.append(temp)
    rearranged_vs = np.array(rearranged_vs)
    ii = 1
    save_name_dat = directory + fname[:-4] + ".dat"
    with open(save_name_dat,'w') as f1:
        for line in rearranged_vs:
            if ii == 3:
                ii = 1
                #f.write('\n')
                f1.write(' '.join([str(x) for x in line.flatten()]))
                f1.write('\n')
                f1.write('\n')
            else:
                f1.write(' '.join([str(x) for x in line.flatten()]))
                f1.write('\n')
                ii += 1

def render_surfaces_gnuplot(fname, directory,
                            V, F,
                            extra_field = []):
    """
    # Source; http://gnuplot.info/demo_5.4/polygons.3.gnu
    # Color: https://louisem.com/417898/blue-hex-codes#:~:text=Color%20Name%2C%20Baby%20Blue%2C%20Navy%20Blue%2C%20Steel,128)%2C%20(70%2C%20130%2C%20180)%2C%20(87%2C%20160%2C%20210)%2C    
    """
    # Export DATs
    export_DAT_files_gnuplot(fname, directory, V, F, extra_field)
    # GNUplot files export
    save_name_gp = directory + fname[:-4] + ".asc"
    gp_txt = """
reset
#!/usr/local/bin/gnuplot -persist
# set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
#set term svg
#set output "{0}"
unset border
unset key
set style line 101  linecolor rgb "forest-green"  linewidth 1.000 dashtype solid pointtype 101 pointsize default
set style line 102  linecolor rgb "goldenrod"  linewidth 1.000 dashtype solid pointtype 102 pointsize default
set view 30, 30, 1.5, 1
set view  equal xyz
set style data lines
unset xtics
unset ytics
unset ztics
#unset cbtics
#unset rtics
set title "Stone: fname" 
set xrange [ * : * ] noreverse nowriteback
set x2range [ * : * ] noreverse writeback
set yrange [ * : * ] noreverse nowriteback
set y2range [ * : * ] noreverse writeback
set zrange [ * : * ] noreverse nowriteback
set cbrange [ * : * ] noreverse writeback
set rrange [ * : * ] noreverse writeback
set pm3d depthorder
set pm3d interpolate 1,1 flush begin noftriangles border linecolor rgb "black"  linewidth 1.000 dashtype solid corners2color mean
set pm3d lighting primary 0.5 specular 0.2 spec2 0.1
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
NO_ANIMATION = 1
phi = 1.61803398874989
n = 21
#set border 4095
#splot ".." with polygons fc "gray75"
#splot ".." with polygons fc "gold75"
splot "{1}" with polygons fc rgb "#6693F5"
set view 169, 327
replot
show view
""".format(fname[:-4] + ".svg", fname[:-4] + ".dat")
    f2=open(save_name_gp,'w')
    f2.write(gp_txt)
    f2.write("\n")
    f2.close()
    return;

def read_obj_file(file_path):
    """
    For *.obj files. This one writes GNUplot data files for quad-redering 
    instead of triangles.
    Source: ChatGPT ^_^ + Mahmoud
    """
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertex = list(map(float, parts[1:]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    vertices_array = np.array(vertices)
    # Process the faces list to convert polygons to triangles
    triangles = []
    for face in faces:
        for i in range(1, len(face) - 1):
            triangle = [face[0], face[i], face[i + 1]]
            triangles.append(triangle)
    faces_array = np.array(triangles)
    return vertices_array, faces_array

def render_data_to_tikz(save_name_tex, V, F):
    rearranged_vs = []
    for ff in F:
        for vv in ff:
            rearranged_vs.append(V[vv].tolist())
    rearranged_vs = np.array(rearranged_vs)
    ii = 0
    #save_name_tex = save_name[:-4] + ".dat"
    with open(save_name_tex,'w') as f:
        for line in rearranged_vs:
            ii += 1
            if ii == 4:
                ii = 0
                f.write(' \n')
                f.write(' '.join([str(x) for x in line.flatten()]))
                f.write(' \n')
            else:
                f.write(' '.join([str(x) for x in line.flatten()]))
                f.write(' \n')

if __name__ == "__main__":
    filename = "test_stone.stl"
    render(filename)
    
