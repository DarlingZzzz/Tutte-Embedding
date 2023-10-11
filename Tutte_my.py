import random
import numpy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc
from tqdm import tqdm

# 输入的文件里面不能有空行
def loadVW(object_file):
    with open(object_file) as file:
        line = file.readline().split()
        i = 0
        while True:
            if (line[0] == "#"):
                line = file.readline()


            if (line[0] == "v"):
                line = file.readline().split()
                i += 1

            if (line[0] == "vn"):
                line = file.readline().split()
            if (line[0] == "vt"):
                line = file.readline().split()
            if (line[0] == 'f'):
                break

        W = numpy.zeros((i, i))

        while True:
            t = 0
            if (line[0] == "f"):
                split = line
                W[int(split[1].split("/")[0]) - 1][int(split[2].split("/")[0]) - 1] += 1
                W[int(split[1].split("/")[0]) - 1][int(split[3].split("/")[0]) - 1] += 1
                W[int(split[2].split("/")[0]) - 1][int(split[1].split("/")[0]) - 1] += 1
                W[int(split[2].split("/")[0]) - 1][int(split[3].split("/")[0]) - 1] += 1
                W[int(split[3].split("/")[0]) - 1][int(split[1].split("/")[0]) - 1] += 1
                W[int(split[3].split("/")[0]) - 1][int(split[2].split("/")[0]) - 1] += 1

                W[int(split[3].split("/")[0]) - 1][int(split[3].split("/")[0]) - 1] += 1
                W[int(split[1].split("/")[0]) - 1][int(split[1].split("/")[0]) - 1] += 1
                W[int(split[2].split("/")[0]) - 1][int(split[2].split("/")[0]) - 1] += 1
                line = file.readline().split()
                t += 1
                if line == []:
                    break
            if(line[0] != 'f'and line != []):
                line = file.readline().split()
            if line == []:
                break
        for j in range(0, i):
            W[j][j] = -1 * (W[j][j])
    return W  # V, W, x, y, z


def getContour(W):
    c = W[-1][-1]
    s = set()
    j = -1
    for j in range(len(W)):
        for i in range(0, len(W)):
            if W[j,i] !=0:
                s.add(W[j,i])
                j = i
        if j == len(W)-1:
            break
    return s


def getContourPoints(crsW):
    """

    :param crsW:
    :type crsW : csr_matrix
    :return:
    """
    val = crsW.data     # 稀疏矩阵的数据
    # J = crsW.indices
    rowPTR = crsW.indptr    # 各行的起始值

    s = list()
    # s_bool = [False] * len(len(rowPTR))




    for i in range(0, len(rowPTR) - 1):
        for jj in range(rowPTR[i], rowPTR[i + 1]):
            v = val[jj]
            if v == 1:
                s.append(i)
                break
    return s


def initialXY(W, contourPointSet):

    X = numpy.zeros(len(W))
    Y = numpy.zeros(len(W))


    c = contourPointSet[-1]  #返回pop删除的值并赋值给对象，原列表改变。pop()默认为最后一个元素 删除列表最后一个元素并赋给c
    initial = contourPointSet[-1]

    c_new = -1
    c_error = -1
    over = 'go'

    X = numpy.zeros(len(W))
    Y = numpy.zeros(len(W))

    contour_points_ordered = [c]
    cont_old = []

    # find the longest contour
    for index in range(300):

        while initial not in contour_points_ordered[1:]:
            if (over == 'over'):
                random.shuffle(contourPointSet)  # 打乱边界点
                contour_points_ordered = [initial]

                over = 'go'
                break
            c_old = c
            random.shuffle(contourPointSet)  # 打乱边界点
            for c2 in contourPointSet:
                c2 = int(c2)

                if (W[c][c2] == 1) and (c2 not in contour_points_ordered[1:]) and (c2 != c_error):
                    c = c2
                    contour_points_ordered.append(c2)
                    c_new = c
                    over = 'go'
                    break

            if (c_old == c_new):

                c_error = c_old
                c = contour_points_ordered[-2]
                over = 'over'
                if over == 'over':
                    random.shuffle(contourPointSet)
                for c2 in contourPointSet:
                    c2 = int(c2)
                    a = W[c2, :]

                    if (W[c][c2] == 1) and (c2 not in contour_points_ordered[1:]) and (c2 != c_error):
                        c = c2
                        contour_points_ordered.append(c2)
                        c_new = c

                        over = 'go'
                        break

        if len(contour_points_ordered) > len(cont_old) and len(contour_points_ordered) > 10:
                cont_old = contour_points_ordered
        random.shuffle(contourPointSet)  # 打乱边界点
        initial = contourPointSet[-1]
        contour_points_ordered = [initial]

    # plotting border points on a unit circle
    nbBord = len(cont_old)
    for i in range(0, nbBord-1):

        teta = 2 * numpy.pi * i / (nbBord-1)
        X[cont_old[i]] = numpy.cos(teta)
        Y[cont_old[i]] = numpy.sin(teta)

    return X, Y, cont_old


def buildAndShowNetworkxGraph(crsW, X, Y):
    import networkx as nx

    G = nx.Graph()

    val = crsW.data
    J = crsW.indices
    rowPTR = crsW.indptr
    for i in range(0, len(X)):
        G.add_node(i, pos=(X[i], Y[i]))     # pos 在这

        for jj in range(rowPTR[i], rowPTR[i + 1]):
            j = J[jj]
            if j > i:
                v = val[jj]
                if v > 0:
                    G.add_edge(i, j)

    print("Drawing...")
    nx.draw(G, pos=nx.get_node_attributes(G,'pos'),node_size=4)     # 获取节点G的pos属性
    print("...done drawing")
    print("Showing...")
    plt.show()
    print("...done showing")


import sys
import numpy as np


# 分解矩阵
def DLU(A):
    D = np.zeros(np.shape(A))
    L = np.zeros(np.shape(A))
    U = np.zeros(np.shape(A))
    for i in range(A.shape[0]):
        D[i, i] = A[i, i]
        for j in range(i):
            L[i, j] = -A[i, j]
        for k in list(range(i + 1, A.shape[1])):
            U[i, k] = -A[i, k]
    L = np.mat(L)
    D = np.mat(D)
    U = np.mat(U)
    return D, L, U


# 迭代
def Jacobi_iterative(A, b, x0, maxN, p):  # x0为初始值，maxN为最大迭代次数，p为允许误差
    D, L, U = DLU(A)
    if len(A) == len(b):
        M = np.linalg.inv(D - L)
        D_inv = np.mat(M)
        B = D_inv * U
        B = np.mat(B)
        f = M * b
        f = np.mat(f)
    else:
        print('维数不一致')
        sys.exit(0)  # 强制退出

    a, b = np.linalg.eig(B)  # a为特征值集合，b为特征值向量
    c = np.max(np.abs(a))  # 返回谱半径
    if c < 1:
        print('迭代收敛')
    else:
        print('迭代不收敛')
        # sys.exit(0)  # 强制退出
    # 以上都是判断迭代能否进行的过程，下面正式迭代
    k = 0
    while k < maxN:
        x = B * x0 + f
        k = k + 1
        eps = np.linalg.norm(x - x0, ord=2)
        if eps < p:
            break
        else:
            x0 = x
    return k, x

if __name__ == '__main__':

    object_file_path = "bunny_10k_close.obj"
    W = loadVW(object_file_path)
    print("VW creation DONE")

    crsW = sp.csr_matrix(W)
    print("W Sparsing DONE")
    contour_points_set = getContourPoints(crsW)
    print("contour_points_set",contour_points_set)

    X, Y, contour_pts_ordered = initialXY(W, contour_points_set)

    print(len(contour_pts_ordered),contour_pts_ordered)
    contour_points = set(contour_pts_ordered)
    other_points = set(range(0,len(W)))-contour_points
    print("CONTOUR DONE")


    W_copy = W
    X_copy = X
    Y_copy = Y
    # 修改W
    for i in contour_pts_ordered:
        W_copy[i][i] = 1
        for j in range(len(W)):
            if j != i:
                W_copy[i][j] = 0
    for i in range(len(W)):
        for j in range(len(W)):
            if i != j and W_copy[i][j] != 0:
                W_copy[i][j] = 1
    print('Modify DONE')

    A = W_copy
    b = X_copy.reshape(len(X_copy),1)
    c = Y_copy.reshape(len(Y_copy),1)
    x0 = X_copy.reshape(len(X_copy),1)
    y0 = Y_copy.reshape(len(Y_copy),1)
    maxN = 200
    p = 0.0000000001
    k, x = Jacobi_iterative(A, b, x0, maxN, p)
    k, y = Jacobi_iterative(A, c, y0, maxN, p)

    x = np.squeeze(x)
    y = np.squeeze(y)
    x_1 = np.array([])
    y_1 = np.array([])
    for i in range(len(x)):
        x_1 = np.append(x_1,x[i])
        y_1 = np.append(y_1,y[i])


    plt.plot(x_1, y_1, 'r.')
    plt.show()
    plt.figure(5)
    try:
        buildAndShowNetworkxGraph(crsW, x_1, y_1)
    except ImportError:
        print("could not import networkx library, is the networkx python package installed?")


    # import os
    # file_path = 'head_close_map.obj'
    # if not os.path.exists(file_path):
    #     open(file_path, 'w').close()
    #
    # with open('head_close_map.obj', 'a') as f:
    #     for i in range(len(x_1)):
    #         lines = ['v ', '{} '.format(x_1[i]), '{} '.format(y_1[i]),'0']
    #         f.writelines(lines)




