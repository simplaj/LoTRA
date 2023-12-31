import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def calculate_Dij(V):
    """
    Input:
        V: 时延嵌入矩阵，每行为长度为m的子向量
    Output:
        D: Dij距离矩阵，大小为V.shape[0] x V.shape[0]
    """
    n = V.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(V[i,:] - V[j,:])
    return D


def cal_8bit(D, i, j):
    g0 = D[i, j]
    gl = [D[i-1, j-1], D[i-1, j], D[i-1][j+1],
          D[i, j-1], D[i, j+1], D[i+1, j-1], 
          D[i+1, j], D[i+1, j+1]]
    res = [2**k * (1 if gl[k] >= g0 else 0) for k in range(8)]
    return sum(res)


def calculate_Tij(D):
    n = D.shape[0]
    T = np.zeros((n-2, n-2))
    for i in range(1, n-1):
        for j in range(1, n-1):
            T[i-1, j-1] = cal_8bit(D, i, j)
    return T


def process(x, t, m, im_save_path):
    save_dir = os.path.dirname(im_save_path)
    os.makedirs(save_dir, exist_ok=True)
    V = time_delay_embedding(x, t, m)
    D = calculate_Dij(V)
    T = calculate_Tij(D)
    heatmap = cv2.applyColorMap(np.uint8(T), cv2.COLORMAP_JET)
    cv2.imwrite(im_save_path, heatmap)


if __name__ == '__main__':
    x = np.cos(np.linspace(0,10*np.pi,200))
    process(x, 2, 100, './img/cos.jpg')