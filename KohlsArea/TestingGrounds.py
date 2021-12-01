# Author: Kohl Morris
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def getData():
    # Data = pd.read_csv(r"log_2020_08_17-07_28_57_BB.csv")
    Data = pd.read_csv(r"log_2020_09_21-07_14_29_BB.csv")
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
    if len(newVal) > len(Uval):
        min = len(Uval)
    else:
        min = len(newVal)
    for i in range(min):
        if newVal[i].real > 0:
            S[i][i] = math.sqrt(newVal[i].real)
        else:
            S[i][i] = 0
    return newU, S, newV


def centerData(X):
    newX = [[0 for i in range(len(X[0]))] for j in range(len(X))]
    for i in range(len(X[0])):
        sum = 0
        for j in range(len(X)):
           sum = sum + X[j][i]
        mean = sum / len(X)
        for j in range(len(X)):
           newX[j][i] = X[j][i] - mean
        sumDev = 0
        for j in range(len(X)):
           sumDev = sumDev + newX[j][i] ** 2
        stdDev = math.sqrt(sumDev / len(X))
        for j in range(len(X)):
           newX[j][i] = newX[j][i] / stdDev
    return newX


def getFirstCol(X):
    col = []
    for i in range(len(X)):
        col.append(X[i][0])
    return col


if __name__ == '__main__':
    dt = getData()
    data = fixDtNP(pd.DataFrame.to_numpy(dt), "10:30", "10:35")


    X = reconstructX(data[1], 9)
    # centered = centerData(X)
    U, S, V = SVD(X)
    x1 = np.matmul(U, S)
    x = np.matmul(x1, np.transpose(V))
    if(x[0][0] < 0):
        x = x * -1 # '''

    Sd = []
    if len(U) > len(V):
        mini = len(V)
    else:
        mini = len(U)
    for i in range(mini):
        Sd.append(S[i][i])
    totVar = 0 # total variance
    for i in range(len(V)):
        totVar = totVar + Sd[i] * Sd[i]
    totVar = totVar / (len(V) - 1)
    for i in range(len(Sd)):
        Sd[i] = abs(Sd[i] / totVar)
    # plt.yscale('log')
    # plt.plot(Sd, 'o-')
    # plt.show()


    fc = getFirstCol(X) # get the first column
    plt.plot(fc, label='Original')
    tempS = [[0 for i in range(len(V))] for j in range(len(U))]
    mi = min(len(V), len(U))
    for i in range(mi):
        tempS[i][i] = S[i][i]
        xtemp = np.matmul(U, tempS)
        xtemp = np.matmul(xtemp, V.T)
        if xtemp[0][0] < 0:
            xtemp = xtemp * -1
        col = getFirstCol(xtemp)
        string = "" + str(i+1)
        plt.plot(col, label=string)
    plt.legend()
    plt.show() # '''

