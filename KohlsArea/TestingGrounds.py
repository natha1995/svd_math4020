# Author: Kohl Morris
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


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


# reconstruct the phase space given embedding dimension and data
def reconstructX(data, dim):
    X = []
    for i in range(len(data) - (dim - 1)):
        x = []
        for j in range(dim):
            x.append(data[i + j])
        X.append(x)
    return X


def SVD(X):
    npX = np.array(X)
    npXt = np.transpose(npX)

    npXXt = np.matmul(npX, npXt)
    npXtX = np.matmul(npXt, npX)

    Uval, npU = np.linalg.eig(npXXt)
    Vval, npV = np.linalg.eig(npXtX)

    # reorganize U and V
    # selection sort
    Ut = np.ndarray.tolist(np.transpose(npU))
    for i in range(len(Uval) - 1):
        min = i
        for j in range(i+1, len(Uval)):
            if Uval[min] < Uval[j]:
                min = j
        Uval[i], Uval[min] = Uval[min], Uval[i]
        Ut[i], Ut[min] = Ut[min], Ut[i]
    npU = np.transpose(np.array(Ut))

    Vt = np.ndarray.tolist(np.transpose(npV))
    for i in range(len(Vval) - 1):
        min = i
        for j in range(i + 1, len(Vval)):
            if Vval[min] < Vval[j]:
                min = j
        Vval[i], Vval[min] = Vval[min], Vval[i]
        Vt[i], Vt[min] = Vt[min], Vt[i]
    npV = np.transpose(np.array(Vt))

    # Sigma will be denoted as S
    # 1943 * 3
    S = [[0 for i in range(len(Vval))] for j in range(len(Uval))]
    for i in range(len(Vval)):
        S[i][i] = math.sqrt(Vval[i])
    return npU, S, npV




if __name__ == '__main__':
    dt = getData()
    data = fixDtNP(pd.DataFrame.to_numpy(dt), "10:30", "12:30")

    X = reconstructX(data[1], 3)
    U, S, V = SVD(X)
    x = np.matmul(U, S)
    x = np.matmul(x, np.transpose(V))
    # fix x
    fixX = [ [0 for i in range(len(x[0]))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            fixX[i][len(x[0])-1-j] = -1 * x[i][j].real


    print("x")
    print(fixX)
    print("OR:")
    npX = np.array(X)
    print(npX)
    diffX = fixX - npX
    print("Diff")
    print(diffX)


    '''u, s, v = np.linalg.svd(X)
    s1 = [[0 for i in range(len(v))] for j in range(len(u))]
    for i in range(len(s)):
        s1[i][i] = math.sqrt(s[i])
    y = np.matmul(u, s1)
    y = np.matmul(y, v)
    print("y")
    print(y)'''
