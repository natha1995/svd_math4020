# Author: Kohl Morris
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.linalg as sp


def getData():
    Data = pd.read_csv(r"log_2020_08_17-07_28_57_BB.csv")
    return Data


def fixDtNP(data, start, end):
    # create timestamps
    newData = [[], []]
    newData[0].append(datetime.datetime(2020, 8, 17, 7, 28, 57))
    newData[1].append(data[0][1])
    for i in range(len(data) - 1):
        # data[i+1][0] = data.values[i][0] + datetime.timedelta(milliseconds=data.values[i][1])
        newData[1].append(data[i+1][1])
        newData[0].append(newData[0][i] + datetime.timedelta(milliseconds=newData[1][i]))

    # slice
    sHr = int(start[0:2])
    sMi = int(start[3:5])
    eHr = int(end[0:2])
    eMi = int(end[3:5])
    stInd = 0  # start index
    enInd = 10  # end index
    for i in range(len(data)):
        if newData[0][i].hour == sHr and newData[0][i].minute == sMi:
            stInd = i
        if newData[0][i].hour == eHr and newData[0][i].minute == eMi:
            enInd = i
    # return data[stInd:enInd]
    sl1 = newData[0][stInd:enInd]
    sl2 = newData[1][stInd:enInd]

    '''
    # center data
    sum = 0
    for i in range(len(sl2)):
        sum = sum + sl2[i]
    mean = sum / len(sl2)
    for i in range(len(sl2)):
        sl2[i] = sl2[i] - mean # '''

    return [sl1, sl2]


def orgSort():
    '''# reorganize U and V
        # selection sort
        Ut = np.ndarray.tolist(np.transpose(npU))
        for i in range(len(Uval) - 1):
            max = i
            for j in range(i+1, len(Uval)):
                if Uval[max].real < Uval[j].real:
                    max = j
            Uval[i], Uval[max] = Uval[max], Uval[i]
            Ut[i], Ut[max] = Ut[max], Ut[i]
        npU = np.transpose(np.array(Ut))

        Vt = np.ndarray.tolist(np.transpose(npV))
        for i in range(len(Vval) - 1):
            max = i
            for j in range(i + 1, len(Vval)):
                if Vval[max] < Vval[j]:
                    max = j
            Vval[i], Vval[max] = Vval[max], Vval[i]
            Vt[i], Vt[max] = Vt[max], Vt[i]
        npV = np.transpose(np.array(Vt))'''
    print()


# reconstruct the phase space given embedding dimension and data
def reconstructX(data, dim):
    X = []
    for i in range(len(data) - (dim - 1)):
        x = []
        for j in range(dim):
            x.append(data[i + j])
        X.append(x)
    return X


def orthonormalize(vector):
    sumSQ = 0
    for i in range(len(vector)):
        sumSQ = sumSQ + vector[i] * vector[i]
    normal = vector
    for i in range(len(vector)):
        normal[i] = vector[i] / math.sqrt(sumSQ)
    print(normal)
    return normal


def SVD(X):
    npX = np.array(X)
    npXt = np.transpose(npX)

    npXXt = np.matmul(npX, npXt)
    npXtX = np.matmul(npXt, npX)

    Uval, npU = np.linalg.eig(npXXt)
    Vval, npV = np.linalg.eig(npXtX)

    npuR = [[0 for j in range(len(npU[0]))] for i in range(len(npU))]   # the real parts of npU
    for i in range(len(npU)):
        for j in range(len(npU[0])):
            npuR[i][j] = npU[i][j].real
    uValR = [ 0 for i in range(len(Uval))]  # the real parts of Uval
    for i in range(len(uValR)):
        uValR[i] = Uval[i].real

    # Sigma will be denoted as S
    uvs = np.argsort(uValR)     # sort eigenvalues in ascending order
    uflip = np.flip(uvs)        # reverse
    vvs = np.argsort(Vval)      # sort eigenvalues in ascending order
    vflip = np.flip(vvs)        # reverse
    newU = np.transpose(np.array(np.transpose(npuR))[uflip])    # sort
    newV = np.transpose(np.array(np.transpose(npV))[vflip])     # sort
    newVal = np.array(Vval)[vflip]  # sort


    S = [[0 for i in range(len(Vval))] for j in range(len(Uval))]
    for i in range(len(newVal)):
        S[i][i] = math.sqrt(newVal[i].real)
    return newU, S, newV


if __name__ == '__main__':
    dt = getData()
    data = fixDtNP(pd.DataFrame.to_numpy(dt), "10:30", "10:31")

    X = reconstructX(data[1], 3)
    U, S, V = SVD(X)
    x1 = np.matmul(U, S)
    x = np.matmul(x1, np.transpose(V))
    # print(x)
    # fix x
    fixX = [ [0 for i in range(len(x[0]))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            # fixX[i][len(x[0])-1-j] = -1 * x[i][j].real
            fixX[i][j] = math.floor(x[i][j].real)
    u, s, vt = np.linalg.svd(X, full_matrices=True)
    print()

    '''print("x")
    print(x)
    print("OR:")
    npX = np.array(X)
    print(npX)  # '''

