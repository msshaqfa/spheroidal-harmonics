"""
%*************************************************************************%
% All rights reserved (C) to the authors: Mahmoud SHAQFA,                 %
%                  Ketson dos SANTOS & and Katrin BEYER                   %
%                                                                         %
% M. Shaqfa Contact:                                                      %
% Department of Mechanical Engineering, Massachusetts Institute of        %
% Technology (MIT)                                                        %
% Cambridge, MA, USA                                                      %
%               Email: mshaqfa@mit.edu                                    %
%                                                                         %
% K. dos Santos Contact:                                                  %
% Department of Civil, Environmental, and Geo- Engineering, University    %
% of Minnesota, 500 Pillsbury Drive S.E., Minneapolis, MN 55455-0116 USA  %
%               Email: dossantk@umn.edu                                   %
%                                                                         %
% K. Beyer Contact:                                                       %
%               Email: katrin.beyer@epfl.ch                               %
%*************************************************************************%
% This code includes implementations for:                                 %
%				- Disk harmonics expansion for parametric surfaces        %
% This code is part of the paper: "Disk Harmonics for Analysing Curved    %
% and Flat Self-affine Rough Surfaces and the Topological                 %
% Reconstruction of Open Surfaces"                                        %
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
Authors of this file: Mahmoud S. Shaqfa and Ketson dos Santos
EPFL-EESD @2022
"""

import time
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import interpolate
from scipy import linalg as lng

import pyvista as pv

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from joblib import parallel_backend
import cvxpy as cp

# Read databases
import pickle
import h5py
import scipy.io as sio
import mat73

class HarmIO:
    """
        Class for reading and writing hdf5 files.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def read_hdf5(filename = "test.hdf5"):
    # Matlab v7.3 *mat files are simply hdf5 files only export 7.3!
        f = h5py.File(filename,'r')
        v = np.float64(np.array(f.get('v')))
        keys = list(f.keys()) 
        indices = [i for i, s in enumerate(keys) if 'mat' in s]
        D_mat = f.get(str(keys[indices[0]]))
        D_mat = np.complex64(D_mat['real']+D_mat['imag']*1j)
        return v, D_mat
    
    @staticmethod
    def export_mat(filename = "test.mat", dict={}):
        # Export Matlab v7.3 *.mat files
        """
            example:    dicts = {"qm_k": coefs}
                        export_mat("test.mat", dict)
        """
        sio.savemat(filename, dict)
    
    @staticmethod
    def export_variables(fname, dict):
        with open(fname, 'wb') as file:  
            pickle.dump(dict, file)
    
    @staticmethod
    def import_variables(fname):
        with open(fname, 'rb') as file:
            myvar = pickle.load(file)
        return myvar

class RandomChoice:
    """
        A fast rando m sampler that replaces the np.random.choice().
    """
    def __init__(self, n, p=None, random_state=None):
        self.n = np.array(n)
        self.p = np.array(p)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        if p is None:
            p = np.ones(n.shape)/len(n)
        idx = np.arange(len(p))[::1]
        psort=np.sort(p)
        parg=np.argsort(p)
        self.n_sort = self.n[parg]
        # Interpolator of the inverse CDF (continuous approximation).
        csort = np.cumsum(psort)
        self.inv_cdf = interpolate.interp1d(csort, idx)
        self.min_c = min(csort)
        self.max_c = max(csort)
    def get(self):
        # Choose a random number from the CDF.
        u = (self.max_c-self.min_c)*self.rng.random() + self.min_c
        # Sampling 'n'.
        sample = self.inv_cdf(u)
        sample = self.n_sort[int(np.round(sample))]
        return sample
        
    
class L2Solvers:
    def __init__(self):
        pass
    
    @staticmethod
    def solve_2_norm(D_mat, v, printed=True):
        # The fastest one of them (parallel computation LAPACK)
        t1 = time.time()
        coefs = np.linalg.lstsq(D_mat, v, rcond=None)
        t2 = time.time()-t1
        if printed: print("\nSolver's time: {} min".format(str(t2/60)))
        coefs = np.asarray(coefs[0], dtype=np.complex64) # Watch out the data type
        return coefs
    
    @staticmethod
    def solve_p_inv(D_mat, v, printed=True):
        t1 = time.time()
        coefs = np.linalg.pinv(D_mat).dot(v)
        t2 = time.time()-t1
        if printed: print("\nSolver's time: {} min".format(str(t2/60)))
        return coefs
    
    @staticmethod
    def solve_2_norm_Ridge(D_mat, v, alpha=0.1, printed=True):
        t1 = time.time()
        X = []
        N = np.shape(D_mat)[1]
        for i in range(0, N):
            X.append(np.real(D_mat[:,i]))
            X.append(np.imag(D_mat[:,i]))
        X = np.array(X).T
        clf = Ridge(alpha)
        clf.fit(X, v)
        coefs = clf.coef_
        t2 = time.time()-t1
        if printed: print("\nSolver's time: {} min".format(str(t2/60)))
        return coefs
    
    @staticmethod
    def solve_2_norm_inverse_proj(D_mat, v, printed=True):
        # very slow if not parallel
        t1 = time.time()
        coefs = (np.linalg.inv(np.transpose(D_mat).dot(D_mat)).dot(np.transpose(D_mat))).dot(v)
        t2 = time.time()-t1
        if printed: print("\nSolver's time: {} min".format(str(t2/60)))
        return coefs
    
    @staticmethod
    def solve_2_norm_QR(D_mat, v, printed=True):
        t1 = time.time()
        q, r = np.linalg.qr(D_mat)
        coefs = np.dot(np.linalg.pinv(r), np.dot(q.T, v))
        t2 = time.time()-t1
        if printed: print("\nSolver's time: {} min".format(str(t2/60)))
        return coefs


class L1Solver:
    def __init__(self):
        pass

    @staticmethod
    def solve_1_norm_Lasso(D_mat, v, alpha=0.1):
        t1 = time.time()
        X = []
        N = np.shape(D_mat)[1]
        for i in range(0, N):
            X.append(np.real(D_mat[:,i]))
            X.append(np.imag(D_mat[:,i]))
        X = np.array(X).T
        clf = Lasso(alpha)
        clf.fit(D_mat, v)
        coefs = clf.coef_
        t2 = time.time()-t1
        print("\nSolver's time: {} min".format(str(t2/60)))
        return coefs

class visulaiser:
    def __init__(self):
        pass
    
    def plot_pointCloud(v, map="r"):
        cloud = pv.PolyData(v)
        if map=="r":
            cloud['point_color'] = np.sqrt(cloud.points[:, 0]**2 + cloud.points[:, 1]**2 + cloud.points[:, 2]**2)  # just use radial coordinate
        elif map=="z":
            cloud['point_color'] = cloud.points[:, 2]  # just use z coordinate
        pv.plot(cloud, scalars='point_color', cmap='jet', show_bounds=True, cpos='yz', show_scalar_bar=True)
        pv.plotting._ALL_PLOTTERS.clear()



def randomised_sampler(D_mat, v, support_size):
    # This is to retrun a random sample of the equations for sparse solutions (assuming a support size)
    ip = np.random.permutation(len(D_mat[:,1]))
    idel = ip[:support_size]
    D0 = D_mat[idel,:]
    V0 = v[idel,:]
    return D0, V0



# Reconstruction function
def reconstruction_with_basis(D_mat, coefs, n_max):
    n_max = np.min([n_max, int(np.sqrt(D_mat.shape[0])-1)])
    rec_v = np.dot(D_mat[0:(n_max+1)**2, :].T, coefs[0:(n_max+1)**2,:])
    return rec_v.real

# Functions to separate the postive and zero orders from the negative orders
def classify_coefs(coefs):
    """
        This function will seperate the coefficients to negatives only or postive and zeros.
    """
    k_max = int(np.sqrt(coefs.shape[0])-1)
    pos_zero_coefs = np.zeros((int(((k_max+1)*(k_max+2)*0.5)),), dtype=np.complex64)
    neg_coefs = np.zeros((int(k_max*(k_max+1)*0.5),), dtype=np.complex64)
    for _k in range(k_max+1):
        for _m in range(-_k, _k+1):
            if _m >= 0:
                pos_zero_coefs[int(_k*(_k+1)*0.5 + _m)] = coefs[_k**2 +_k + _m]
            else:
                neg_coefs[int(_k*(_k-1)*0.5 + abs(_m)-1)] = coefs[_k**2 +_k + _m]
    return pos_zero_coefs, neg_coefs

def classify_order_basis(bases):
    """
        This function will calssify the bases into positive and zero orders only and negative orders only.
        The bases matrix should be full-rank column matrix, with n rows (points) and beta columns (bases).
    """
    k_max = int(np.sqrt(bases.shape[1])-1)
    pos_zero_bases = np.zeros((bases.shape[0], int(((k_max+1)*(k_max+2)*0.5))), dtype=np.complex64)
    neg_bases = np.zeros((bases.shape[0], int(k_max*(k_max+1)*0.5)), dtype=np.complex64)
    for _k in range(k_max+1):
        for _m in range(-_k, _k+1):
            if _m >= 0:
                pos_zero_bases[:, int(_k*(_k+1)*0.5 + _m)] = bases[:, _k**2 +_k + _m]
            else:
                neg_bases[:, int(_k*(_k-1)*0.5 + abs(_m)-1)] = bases[:, _k**2 +_k + _m]
    return pos_zero_bases, neg_bases

def declassify_coefs(sym_coefs):
    """
        This function will map the postive and zero coefficients to negative, zero and positive coefs (recover the original system).
    """
    k_max = int(-1.5 + np.sqrt(0.25 + 2 * sym_coefs.shape[0]))    
    coefs = np.zeros((int( (k_max+1) **2 ),), dtype=np.complex64)
    for _k in range(k_max+1):
        for _m in range(0, _k+1):
            coefs[_k**2 +_k + _m ] = sym_coefs[int(_k*(_k+1)*0.5 + _m)]
    
    for _k in range(k_max+1):
        for _m in range(-_k, 0):
            coefs[_k**2 +_k + _m] = (-1)**abs(_m) * coefs[_k**2 +_k + abs(_m)].conj()    
    return coefs


def symmetry_indexing_system(k_max):
    """
        Return the signs associated with conjugate symmetry bases (indexing systems).
    """
    _sym_signs = []
    for _k in range(k_max+1):
        for _m in range(1, _k+1):
            _sym_signs.append( int((-1)**(_m)) )
    _sym_signs = np.array(_sym_signs)
    
    # Find the indicies of the positive coefs only
    _pos_idx = []
    for _k in range(k_max+1):
        for _m in range(1, _k+1):
            _pos_idx.append( int(_k*(_k+1)*0.5 + _m) )
    _pos_idx = np.array(_pos_idx)
    return _sym_signs, _pos_idx

# Implementation of Kaczmarz algorithmns
class Kaczmarz_solver:
    def __init__(self, D_mat, v, x0=[], iterations = 3, relaxation = 0.3):
        self.D_mat = D_mat
        self.v = v
        self.relaxation = relaxation
        self.m, self.n = D_mat.shape
        self.k_max = int(np.sqrt(self.n)-1) # Max expansion degrees embeded in the bases
        self.iterations = int(iterations * self.m) # a simple multiplier
        if list(x0) != []:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.n)
        
        # Precompute processes for less overhead
        self.P = (np.linalg.norm(D_mat, axis=1) / np.linalg.norm(D_mat))**2.0
        self.rows_norm = np.linalg.norm(D_mat, axis=1)**2
        self.conj_D_mat = self.D_mat.conj()
        
        self.solutions_history = np.zeros((self.iterations+1, self.n), dtype=np.complex64)
        self.best_solution = []
        self.best_parallel_solution = []
        self.sym_switch = False
        self.history_switch = False
        
    def solve_Kaczmarz(self):
        """
            For overdetermined system use strong underelaxation with relaxation = 0.3
            For underdetermined systems use strong overelaxation with relaxation = 1.2-1.3
            For relaxation = 1, we use the standard Kaczmarz method.
        """
        self.sym_switch = False
        self.solutions_history = np.zeros((self.iterations+2, self.n), dtype=np.complex64)
        t1 = time.time()
        k = 0
        X = self.x0
        self.solutions_history[k, :] = X
        random_choice = RandomChoice(n = range(self.m), p = self.P)
        while k <= self.iterations:
            i = random_choice.get()
            Xnew = X + self.relaxation * (self.v[i] - self.D_mat[i,:] @ X) / self.rows_norm[i] * self.conj_D_mat[i,:]
            X = Xnew
            k += 1
            self.solutions_history[k, :] = X
        print("\nSolver's time: {} sec".format(str((time.time()-t1))))
        self.best_solution = X
        self.history_switch = True
        return X
    
    def solve_Kaczmarz_parallel(self, label="coefs", label_hist="history", return_dict=[]):
        """
            This is a solver manager for x, y and z coefs with parallel RKaczmarz.
        """
        return_dict[label] = self.solve_Kaczmarz()
        return_dict[label_hist] = self.solutions_history
    
    # Assuming conjugate symmetry
    def solve_Kaczmarz_conjugate_sym(self):
        """
            Randmoized Kaczmarz solver assuming the conjugate symmetry of the provided bases.
            This solver is more efficient and faster in converegence as we reduced the number of unkowns.
        """
        self.sym_switch = True
        # Classify bases and find indicies
        _pos_zer_D_mat, _neg_D_mat = classify_order_basis(self.D_mat)
        _sym_signs, _pos_idx = symmetry_indexing_system(self.k_max)
        self.conj_D_mat = _pos_zer_D_mat.conj()
        self.m, self.n = _pos_zer_D_mat.shape
        self.solutions_history = np.zeros((self.iterations+2, self.n), dtype=np.complex64)
        # Re-Precompute processes for less overhead and the modified postive and zero order bases
        self.P = (np.linalg.norm(_pos_zer_D_mat, axis=1) / np.linalg.norm(_pos_zer_D_mat))**2.0
        self.rows_norm = np.linalg.norm(_pos_zer_D_mat, axis=1)**2
        
        t1 = time.time()
        k = 0
        X, _ = classify_coefs(self.x0)
        self.solutions_history[k, :] = X
        random_choice = RandomChoice(n = range(self.m), p = self.P)
        while k <= self.iterations:
            i = random_choice.get()
            _v_hat = self.v[i] - _neg_D_mat[i,:] @ (X[_pos_idx].conj() * _sym_signs)
            Xnew = X + self.relaxation * (_v_hat - _pos_zer_D_mat[i,:] @ X) / self.rows_norm[i] * self.conj_D_mat[i,:]
            X = Xnew
            k += 1
            self.solutions_history[k, :] = X
        X = declassify_coefs(X)
        self.best_solution = X
        self.history_switch = True
        print("\nSolver's time: {} sec".format(str((time.time()-t1))))
        return X
    
    def solve_Kaczmarz_conjugate_sym_parallel(self, label_coefs="coefs", label_hist="history", return_dict=[]):
        """
            Randmoized Kaczmarz solver assuming the conjugate symmetry of the provided bases.
            This solver is the same as the conj sym solver, but in parallel.
        """
        return_dict[label_coefs] = self.solve_Kaczmarz_conjugate_sym()
        return_dict[label_hist] = self.solutions_history
    
    # Assuming sparse system
    # Assuming conjugate symmetry
    def solve_Kaczmarz_sparse_conj_sym(self, support_sz):
        """
            Randmoized Kaczmarz solver assuming the bases are sparse with a given support size (support_sz).
            This solver is more efficient and faster in converegence as we reduced the number of unkowns.
        """
        self.sym_switch = True
        # Classify bases and find indicies
        _pos_zer_D_mat, _neg_D_mat = classify_order_basis(self.D_mat)
        _sym_signs, _pos_idx = symmetry_indexing_system(self.k_max)
        #self.conj_D_mat = _pos_zer_D_mat.conj()
        self.m, self.n = _pos_zer_D_mat.shape
        self.solutions_history = np.zeros((self.iterations+2, self.n), dtype=np.complex64)
        
        t1 = time.time()
        k = 0
        X, _ = classify_coefs(self.x0)
        self.solutions_history[k, :] = X
        random_choice = RandomChoice(n = range(self.m), p = self.P)
        while k <= self.iterations:
            i = random_choice.get()
            _num = max(support_sz, self.n - k)
            lx = len(X)
            w = np.ones(lx)
            if _num < self.n:
                Sc = abs(X).argsort()[-lx:][lx-_num-1::-1]
                w[np.array(Sc)]=1/np.sqrt(k+1)
            else:
                Sc = []
            _v_hat = self.v[i] - (w[_pos_idx] * _neg_D_mat[i,:]) @ (X[_pos_idx].conj() * _sym_signs)
            Xnew = X + self.relaxation * (_v_hat - (w * _pos_zer_D_mat[i,:]) @ X) /\
                np.linalg.norm(w * _pos_zer_D_mat[i,:])**2 * (w * _pos_zer_D_mat[i,:]).conj()
            X = Xnew
            k += 1
            self.solutions_history[k, :] = X
        X = declassify_coefs(X)
        self.best_solution = X
        self.history_switch = True
        print("\nSolver's time: {} sec".format(str((time.time()-t1))))
        return X
        
    def solve_Kaczmarz_sparse_conj_sym_parallel(self, support_sz, label_coefs="coefs", label_hist="history", return_dict=[]):
        """
            Randmoized Kaczmarz solver assuming the bases are sparse with a given support size (support_sz).
            This is the same as solve_Kaczmarz_sparse_conj_sym but with parallel shared memeory.
        """
        return_dict[label_coefs] = self.solve_Kaczmarz_sparse_conj_sym(support_sz)
        return_dict[label_hist] = self.solutions_history
        
    
    def solve_Kaczmarz_sparse(self, support_sz):
        """
            Randmoized Kaczmarz solver assuming the bases are sparse with a given support size (support_sz).
            This solver is more efficient and faster in converegence as we reduced the number of unkowns.
        """
        self.sym_switch = False
        self.solutions_history = np.zeros((self.iterations+2, self.n), dtype=np.complex64)
        t1 = time.time()
        k = 0
        X = self.x0
        self.solutions_history[k, :] = X
        random_choice = RandomChoice(n = range(self.m), p = self.P)
        while k <= self.iterations:
            i = random_choice.get()
            _num = max(support_sz, self.n - k)
            lx = len(X)
            w = np.ones(lx)
            if _num < self.n:
                Sc = abs(X).argsort()[-lx:][lx-_num-1::-1]
                w[np.array(Sc)]=1/np.sqrt(k+1)
            else:
                Sc = []
            Xnew = X + self.relaxation * (self.v[i] - (w * self.D_mat[i,:]) @ X) /\
                np.linalg.norm(w * self.D_mat[i,:])**2 * (w * self.D_mat[i,:]).conj()
            X = Xnew
            k += 1
            self.solutions_history[k, :] = X
        print("\nSolver's time: {} sec".format(str((time.time()-t1))))
        self.best_solution = X
        self.history_switch = True
        return X
    
    def solve_Kaczmarz_sparse_parallel(self, support_sz, label_coefs="coefs", label_hist="history", return_dict=[]):
        """
            Randmoized Kaczmarz solver assuming the bases are sparse with a given support size (support_sz).
            This solver is more efficient and faster in converegence as we reduced the number of unkowns.
            This is the same as solve_Kaczmarz_sparse but with parallel shared memeory.
        """
        return_dict[label_coefs] = self.solve_Kaczmarz_sparse(support_sz)
        return_dict[label_hist] = self.solutions_history
    
    
        # Assuming conjugate symmetry
    def cont_solve_Kaczmarz_conjugate_sym(self):
        """
            Randmoized Kaczmarz solver assuming the conjugate symmetry of the provided bases.
            This solver is more efficient and faster in converegence as we reduced the number of unkowns.
        """
        self.sym_switch = True
        # Classify bases and find indicies
        _pos_zer_D_mat, _neg_D_mat = classify_order_basis(self.D_mat)
        _sym_signs, _pos_idx = symmetry_indexing_system(self.k_max)
        self.conj_D_mat = _pos_zer_D_mat.conj()
        self.m, self.n = _pos_zer_D_mat.shape
        self.solutions_history = np.zeros((self.iterations+2, self.n), dtype=np.complex64)
        # Re-Precompute processes for less overhead and the modified postive and zero order bases
        self.P = (np.linalg.norm(_pos_zer_D_mat, axis=1) / np.linalg.norm(_pos_zer_D_mat))**2.0
        self.rows_norm = np.linalg.norm(_pos_zer_D_mat, axis=1)**2
        
        t1 = time.time()
        k = 0
        X, _ = classify_coefs(self.x0)
        self.solutions_history[k, :] = X
        random_choice = RandomChoice(n = range(self.m), p = self.P)
        while k <= self.iterations:
            i = random_choice.get()
            _v_hat = self.v[i] - _neg_D_mat[i,:] @ (X[_pos_idx].conj() * _sym_signs)
            Xnew = X + self.relaxation * (_v_hat - _pos_zer_D_mat[i,:] @ X) / self.rows_norm[i] * self.conj_D_mat[i,:]
            X = Xnew
            k += 1
            self.solutions_history[k, :] = X
        X = declassify_coefs(X)
        self.best_solution = X
        self.history_switch = True
        print("\nSolver's time: {} sec".format(str((time.time()-t1))))
        return X
    
    def reconstruct_history(self, rec_steps=10000):
        """
            This function is called after the solver.
            It is used for reconstructing the average error in reconstruction
        """
        calc_rec_error = lambda Xs: np.sqrt(np.mean(( (self.D_mat @ Xs).real - self.v)**2.0))
        
        self.error_history = []
        if self.history_switch:
            for k in range(self.solutions_history.shape[0]):
                if k % rec_steps == 0:
                    if self.sym_switch:
                        self.error_history.append(calc_rec_error(declassify_coefs(self.solutions_history[k, :])))
                    else:
                        self.error_history.append(calc_rec_error(self.solutions_history[k, :]))
        else:
            print("Empty history! Please try to run the solver first.")
        return np.array(self.error_history)
    
    def reconstruct_history_parallel(self, rec_steps=10000, label="coefs", return_dict=[]):
        """
            Parallel error reconstruction.
        """
        return_dict[label] = self.reconstruct_history(rec_steps = rec_steps)