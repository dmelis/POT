# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein transport method - ported from POT and made cupy-based. 
Please refer to ot.gromov in POT package for function documentation.
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#
# Ported to Cupy: David Alvarez Melis <dalvmel@mit.edu>

import cupy as np  # np used for matrix computation
import cupy as cp  # cp used for cupy specific operations
from . import utils

def tensor_product(constC, hC1, hC2, T):
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    return tens

def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)
    return np.sum(tens * T)
    
def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2, T)  # [12] Prop. 2 misses a 2 factor



### CUDAMAT
# def init_matrix(C1, C2, T, p, q, loss_fun='square_loss'):
#     """
#         GPU version of ot.gromov.init_matrix
#     """
#     if loss_fun == 'square_loss':
#         def f1(a):
#             out = cm.empty(a.shape)
#             cm.pow(a,2, target=out)
#             return out.divide(2)
#         def f2(b):
#             out = cm.empty(b.shape)
#             cm.pow(b,2, target=out)
#             return out.divide(2)
#         def h1(a):
#             return a.copy()
#         def h2(b):
#             return b.copy()
#     elif loss_fun == 'kl_loss':
#         def f1(a):
#             return a * np.log(a + 1e-15) - a
#         def f2(b):
#             return b
#         def h1(a):
#             return a
#         def h2(b):
#             return np.log(b + 1e-15)
#
#     p_GPU  = cm.CUDAMatrix(p.reshape(-1, 1))
#     q_GPU  = cm.CUDAMatrix(q.reshape(1, -1))
#
#     ones_q = cm.empty((1, len(q)))
#     ones_q.assign(1)
#     ones_p = cm.empty((len(p), 1))
#     ones_p.assign(1)
#     #ones_q = cm.CUDAMatrix(np.ones(len(q)).reshape(1,-1))
#     #ones_p = cm.CUDAMatrix(np.ones(len(p)).reshape(-1,1))
#
#     constC1 = cm.dot(cm.dot(f1(C1), p_GPU), ones_q)    # (m x 1) * (1 x m)
#     constC2 = cm.dot(ones_p, cm.dot(q_GPU, f2(C2).transpose()))
#     constC1.add(constC2)
#     hC1 = h1(C1)
#     hC2 = h2(C2)
#     return constC1, hC1, hC2
#
# def tensor_product(constC, hC1, hC2, T):
#     """
#         GPU version of ot.gromov.tensor_product
#     """
#     A = cm.dot(cm.dot(hC1, T),hC2.transpose())
#     A.mult(-1)
#     tens = cm.empty(constC.shape)
#     constC.add(A, target = tens)
#     # tens -= tens.min()
#     return tens
#
# def gwloss(constC, hC1, hC2, T):
#     """
#         GPU version of ot.gromov.gwloss
#     """
#     tens = tensor_product(constC, hC1, hC2, T)
#     return tens.mult(T).asarray().sum()
#
#
# def gwggrad(constC, hC1, hC2, T):
#     """
#         GPU version of ot.gromov.gwgrad
#     """
#     G = tensor_product(constC, hC1, hC2, T)
#     G.mult(2) # [12] Prop. 2 misses a 2 factor
#     return G
#
    
# def cosine_distance_gpu(X, Y, returnAsGPU=False, squared=False):
#     """
#     Compute the pairwise euclidean distance between matrices a and b.
#     Parameters
#     ----------
#     a : np.ndarray (n, f)
#         first matrice
#     b : np.ndarray (m, f)
#         second matrice
#     returnAsGPU : boolean, optional (default False)
#         if True, returns cudamat matrix still on GPU, else return np.ndarray
#     squared : boolean, optional (default False)
#         if True, return squared euclidean distance matrice
#     Returns
#     -------
#     c : (n x m) np.ndarray or cudamat.CUDAMatrix
#         pairwise euclidean distance distance matrix
#     """
#     # a is shape (n, f) and b shape (m, f). Return matrix c of shape (n, m).
#     # First compute in c_GPU the squared euclidean distance. And return its
#     # square root. At each cell [i,j] of c, we want to have
#     # sum{k in range(f)} ( (a[i,k] - b[j,k])^2 ). We know that
#     # (a-b)^2 = a^2 -2ab +b^2. Thus we want to have in each cell of c:
#     # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2).
#
#     X_GPU = cm.CUDAMatrix(X)
#     Y_GPU = cm.CUDAMatrix(Y)
#
#     # Gram matrix
#     c_GPU = cm.dot(X_GPU, Y_GPU.transpose()) # n x m
#
#
#     # Compute Norms
#     X_GPU = cm.pow(X_GPU, 2).sum(axis=1) # n x 1
#     Y_GPU = cm.pow(Y_GPU, 2).sum(axis=1) # m x 1
#     c_GPU.div_by_col(cm.sqrt(X_GPU))
#     c_GPU.div_by_row(cm.sqrt(Y_GPU).transpose())
#     c_GPU.mult(-1)
#     c_GPU.add(1)  # 1 - <>/|| |||| ||
#     if returnAsGPU:
#         return c_GPU
#     else:
#         return c_GPU.asarray()
