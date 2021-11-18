import numpy as np
import matplotlib.pyplot as plt


# currently random
def getData():
    x = [np.random.randn() for i in range(1000)]
    return x


# reconstruct the phase space given embedding dimension and data
def reconstructX(data, dim):
    X = []
    for i in range(len(data) - (dim - 1)):
        x = []
        for j in range(dim):
            x.append(data[i + j])
        X.append(x)
    return X


if __name__ == '__main__':
    print("hi")
    X = reconstructX(getData(), 3)
    print(X)