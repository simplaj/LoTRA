import numpy as np


def time_delay_embedding(x, m, t):
    """
    Input:
        x: 一维时间序列数据
        m: 嵌入维度
        t: 嵌入延迟
    Output:
        V: 时延嵌入矩阵，每行为长度为m的子向量
    """
    n = len(x)
    V = np.zeros((n - (m-1)*t, m))
    for i in range(n - (m-1)*t):
        V[i,:] = x[i:i+m*t:t]
    return V


def calculate_Dij(V, x):
    """
    Input:
        V: 时延嵌入矩阵，每行为长度为m的子向量
        x: 一维时间序列数据
    Output:
        D: Dij距离矩阵，大小为V.shape[0] x V.shape[0]
    """
    n = len(x)
    D = np.zeros((V.shape[0], V.shape[0]))
    for i in range(V.shape[0]):
        for j in range(V.shape[0]):
            D[i,j] = np.linalg.norm(k * V[i,:] - V[j,:])
    return D


def cal_8bit(D, i, j):
    g0 = D[i, j]
    gl = [D[i-1, j-1], D[i-1, j], D[i-1][j+1],
          D[i, j-1], D[i, j+1], D[i+1, j-1], 
          D[i+1, j], D[i+1, j+1]]
    res = [2**i * (1 if gl[i] >= g0 else 0) for i in range(8)]
    return sum(res)


def calculate_Tij(D):
    n = d.shape[0]
    T = np.zeros((n-2, n-2))
    for i in range(1, n-1):
        for j in range(1, n-1):
            T[i, j] = cal_8bit(D, i, j)
    return T