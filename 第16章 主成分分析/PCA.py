#PCA（principal components analysis）即主成分分析技术旨在利用降维的思想，把多指标转化为少数几个综合指标。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
data = loadmat('data/ex7data1.mat')
# data
X = data['X']

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

U, S, V = pca(X)
print(U, S, V)
#现在我们有主成分（矩阵U），我们可以用这些来将原始数据投影到一个较低维的空间中。
#对于这个任务，我们将实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数。
def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

Z = project_data(X, U, 1)
print(Z)

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)
print(X_recovered)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()
#第一主成分的投影轴基本上是数据集中的对角线。
# 当我们将数据减少到一个维度时，我们失去了该对角线周围的变化，所以在我们的再现中，一切都沿着该对角线。