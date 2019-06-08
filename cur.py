import numpy as np 
import pandas as pd
from scipy import sparse as sp

MAX_MID = 27277 + 1

def select_cols(mat, k, dup=False):
    # prob 1d array of probabilities of all columns
    prob = mat.T.dot(mat)
    prob = np.array(np.diagonal(prob))
    denom = np.abs(prob).sum(axis = 0)
    prob = prob/denom

    C = np.zeros((mat.shape[0], k))
    ind_cols = np.arange(0, prob.size)
    c_ind = []
    for i in range(k):
        rand_sel = np.random.choice(ind_cols, 1, p=prob)
        c_ind.append(rand_sel[0])
        C[:, i] = mat[:, rand_sel[0]]
        C[:, i] = C[:, i]/np.sqrt(k*prob[rand_sel[0]])

    return C, c_ind

def select_rows(mat, k, dup=False):

    prob = mat.dot(mat.T)
    prob = np.array(np.diagonal(prob))
    denom = np.abs(prob).sum(axis=0)
    prob = prob/denom
    print(prob)
    R = np.zeros((k, mat.shape[1]))
    ind_rows = np.arange(0, prob.size)
    r_ind = []
    for i in range(k):
        rand_sel = np.random.choice(ind_rows, 1, p=prob)
        r_ind.append(rand_sel[0])
        R[i, :] = mat[rand_sel[0], :]
        R[i, :] = R[i, :]/np.sqrt(k*prob[rand_sel[0]])
    r_ind = np.array(r_ind)
    return R, r_ind

def matIntersection(mat, c_ind, r_ind):
    
    W = np.zeros((len(r_ind), len(c_ind)))
    for i in range(len(r_ind)):
        W[i] = mat[r_ind[i], c_ind]
    
    return W

def pseudoInverse(W):
    # U = WP (W+)

    # W = X.Z.YT
    X, Z, YT = np.linalg.svd(W)
    
    # W+ = Y.Z+.XT
    XT = X.T
    Y = YT.T
    # Z+ = reciprocal(Z)
    ZP = np.reciprocal(Z)
    ZP = sp.spdiags(ZP, 0, ZP.size, ZP.size)
    ZP = ZP@ZP
    
    # W+ = Y.Z+.XT
    WP = Y@ZP
    WP = WP@XT

    return WP

def main():

    mat = np.array([[0, 1, 2], [2, 0, 4], [1, 3, 2]])
    print(mat)
    C, c_ind = select_cols(mat, 2)
    R, r_ind= select_rows(mat, 2)
    W = matIntersection(mat, c_ind, r_ind)
    U = pseudoInverse(W)
    Pred = C@U@R
    print(Pred)
    print("Finished Execution")

if __name__ == "__main__":
    main()