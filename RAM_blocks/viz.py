import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def viz3(V,flag):
    # V = V / np.max(V)

    x = y = z = t = []
    x1 = y1 = z1 = t1 = []
    x2 = y2 = z2 = t2 = []
    x3 = y3 = z3 = t3 = []
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                if V[i, j, k] > 0.05 * V.max():
                    x = x + [i]
                    y = y + [j]
                    z = z + [k]
                    t = t + [V[i, j, k]]
                    if i == V.shape[0] / 2:
                        y1 = y1 + [j]
                        z1 = z1 + [k]
                        t1 = t1 + [V[i, j, k]]
                    if j == V.shape[1] / 2:
                        x2 = x2 + [i]
                        z2 = z2 + [k]
                        t2 = t2 + [V[i, j, k]]
                    if k == V.shape[2] / 2:
                        x3 = x3 + [i]
                        y3 = y3 + [j]
                        t3 = t3 + [V[i, j, k]]

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    t = np.asarray(t)
    y1 = np.asarray(y1)
    z1 = np.asarray(z1)
    t1 = np.asarray(t1)
    x2 = np.asarray(x2)
    z2 = np.asarray(z2)
    t2 = np.asarray(t2)
    x3 = np.asarray(x3)
    y3 = np.asarray(y3)
    t3 = np.asarray(t3)

    if flag == 0:
        return x,y,z,t

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    im = ax.scatter(x, y, z, c=t, marker='o', s=10)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])
    ax.set_zlim(0, V.shape[2])

    ax = fig.add_subplot(222)
    im = ax.scatter(y1, z1, c=t1, marker='o', s=30)
    ax.set_xlabel('Y Label')
    ax.set_ylabel('Z Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])

    ax = fig.add_subplot(223)
    im = ax.scatter(x2, z2, c=t2, marker='o', s=30)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])

    ax = fig.add_subplot(224)
    im = ax.scatter(x3, y3, c=t3, marker='o', s=30)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.xlim(0, V.shape[0])
    plt.ylim(0, V.shape[1])

    cax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.show()
    a = 1

def plot_cube(ax, x, y, z, inc, a, i):
    "x y z location and alpha"
    ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z, alpha=a,facecolors='y')
    ax.plot_surface([[x, x + inc], [x, x + inc]], [[y, y], [y + inc, y + inc]], z + inc, alpha=a,facecolors='y')

    ax.plot_surface(x, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
    ax.plot_surface(x + inc, [[y, y], [y + inc, y + inc]], [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

    ax.plot_surface([[x, x], [x + inc, x + inc]], y, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')
    ax.plot_surface([[x, x], [x + inc, x + inc]], y + inc, [[z, z + inc], [z, z + inc]], alpha=a,facecolors='y')

    ax.text(x+inc/2.0, y+inc/2.0, z+inc/2.0, i, fontsize=15)


if __name__ == "__main__":
    WW = np.zeros((10,10,10))
    viz3(WW)
