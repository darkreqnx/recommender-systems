import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse 
from scipy import sparse as sp
import scipy.sparse.linalg as ssl
import csv

# Constants 
MAX_UID = 138493
MAX_MID = 27277 + 1

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=5, 
                                        replace=False)
        train[user, test_ratings] = 0.
        for i in test_ratings:
            test[user, i] = ratings[user, i]
    
    test = sp.csr_matrix(test)
    # Test and training are truly disjoint
    assert(np.sum(test.dot(train.T))==0) 
    return train, test

def sparsity(matrix):
    sparsity = float(len(matrix.nonzero()[0]))
    sparsity /= (matrix.shape[0] * matrix.shape[1])
    sparsity *= 100
    return sparsity

def findSim(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.transpose())  
        sim = sim.toarray()
        sim += epsilon
    elif kind == 'item':
        sim = ratings.transpose().dot(ratings)
        sim = sim.toarray()
        sim += epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def predict(ratings, similarity, kind='user'):
    ratings = ratings.toarray()
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def predict_nobias(ratings, similarity, kind='user'):
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]
        
    return pred

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    print(pred)
    actual = actual[actual.nonzero()].flatten()
    return mse(pred, actual)

def makeSparse():
    urm = np.zeros(shape=(MAX_UID, MAX_MID), dtype=np.float32)
    with open('./data/ratings_c.csv') as trainFile:
        urmReader = csv.reader(trainFile, delimiter=',')
        for row in urmReader:
            if int(row[1]) <= MAX_MID:
                urm[int(row[0])-1, int(row[1])] = float(row[2])

    return sp.csr_matrix(urm, dtype=np.float32)
def main():
    matrix = sp.load_npz('./data/subset_sp.npz')

    train = sp.load_npz('./data/train_subset.npz')
    test = sp.load_npz('./data/test_subset.npz')

    # Dense Matrix
    usim = findSim(matrix, kind="user")
    isim = findSim(matrix, kind="item")

    # Normal CF
    upred = predict(train, usim, kind="user")
    ipred = predict(train, isim, kind="item")

    print('User Based CF : '+str(get_mse(upred, test)))
    print('Item Based CF : '+str(get_mse(ipred, test)))

    # CF with Baseline

    upred1 = predict_nobias(train, usim, kind="user")
    ipred1 = predict_nobias(train, isim , kind="item")

    print('User Based CF : '+str(get_mse(upred1, test)))
    print('Item Based CF : '+str(get_mse(ipred1, test)))

if __name__ == '__main__':
    main()