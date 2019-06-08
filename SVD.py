import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as ssl
import csv
from scipy import spatial
from sparsesvd import sparsesvd
import math as mt
import warnings
warnings.filterwarnings('ignore')

# constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 10000
MAX_UID = 35

FILE_NAME = 'ratings_mod_1000.csv'


def readUrm():
    urm = np.zeros(shape=(MAX_UID + 1, MAX_PID), dtype=np.float32)
    with open(FILE_NAME) as trainFile:
        urmReader = csv.reader(trainFile, delimiter=',')
        count = 0
        for row in urmReader:

            if int(row[2]) <= MAX_PID and int(row[0]) <= MAX_UID:
                urm[int(row[0]), int(row[2])] = float(row[1])
                count += 1
            # else:
            #     break

    print(f'Line Count {count}')
    return csc_matrix(urm, dtype=np.float32)


def readUsersTest():
    uT = dict()
    with open("testuid.csv") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            uT[int(row[0])] = list()

    return uT


def getMoviesSeen():
    moviesSeen = dict()
    with open(FILE_NAME) as trainFile:
        urmReader = csv.reader(trainFile, delimiter=',')
        for row in urmReader:
            try:
                moviesSeen[int(row[0])].append(int(row[2]))
            except:
                moviesSeen[int(row[0])] = list()
                moviesSeen[int(row[0])].append(int(row[2]))

    return moviesSeen


def computeSVD(urm, K):

    U, s, Vt = getSVD(urm, K)

    # np.insert(U, 0, np.array((1, 1)), 0)
    # print(U.shape)
    # print(U)
    # print(s.shape)
    print(s)
    print(Vt.shape)
    print(Vt)
    # print(Vt.transpose())

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        # S[i, i] = mt.sqrt(s[i])
        S[i, i] = s[i]

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def computeSVDpackage(urm, K):

    U, s, Vt = sparsesvd(urm, K)
    # print(U.shape)
    # print(U)
    # print(len(s))
    # print(s.shape)
    print(s)
    # # print(Vt)
    print(Vt.shape)
    print(Vt)
    # # print(Vt.transpose())

    # print(Ux - U)
    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        # S[i, i] = mt.sqrt(s[i])
        S[i, i] = s[i]

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def getSVD(urm, K):

    # SPARSE MATRIX CODE

    A = urm
    AT = A.transpose()

    ATA = AT @ A
    # print(ATA)
    eigs, V = ssl.eigs(ATA, k=MAX_UID)
    # print(eigs)
    eig_vals = []

    for x in eigs:
        if x != 0:
            eig_vals.append(x.real)
    eig_vals = np.array(eig_vals, dtype=np.float32)
    eig_vals[::-1].sort()

    V = V.astype(np.float32)

    VT = V.transpose()
    eig_vals = np.sqrt(eig_vals)

    # print(eig_vals)

    S = np.diag(eig_vals)

    Si = np.linalg.inv(S)

    A = A.todense()
    U = A @ V
    U = U @ Si
    U = U.transpose()
    U = np.negative(U)
    V = np.negative(V)

    return U, eig_vals, VT


# def SVD


def computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, test):
    rightTerm = S * Vt

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm

        # we convert the vector to dense format in order to
        # get the indices of the movies
        # with the best estimated ratings
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
        for r in recom:
            if r not in moviesSeen[userTest]:
                uTest[userTest].append(r)

                if len(uTest[userTest]) == 5:
                    break

    return uTest


def main():
    K = 90

    np.set_printoptions(suppress=True)
    print('Reading Data Set..')
    urm = readUrm()

    print('Reading test user input...')
    users = readUsersTest()
    print('Getting movies already watched by users...')
    moviesSeen = getMoviesSeen()
    print('Computing SVD...')
    U, S, Vt = computeSVD(urm, K)
    uTest = computeEstimatedRatings(urm, U, S, Vt, users, moviesSeen, K, True)

    print(uTest)

    print('Reading test user input...')
    users = readUsersTest()
    print('Getting movies already watched by users...')
    moviesSeen = getMoviesSeen()
    print('Computing SVD with sparseSVD...')
    Up, Sp, Vtp = computeSVDpackage(urm, K)

    uTest = computeEstimatedRatings(
        urm, Up, Sp, Vtp, users, moviesSeen, K, True)

    print(uTest)

    # print(U.shape)
    # print(Up.shape)
    # print(U.todense)
    # print(Up.todense)
    # print(S.shape)
    # print(Sp.shape)
    # print(S.todense)
    # print(Sp.todense)
    # print(Vt.shape)
    # print(Vtp.shape)
    # print(Vt.todense)
    # print(Vt.todense)


if __name__ == '__main__':
    main()
