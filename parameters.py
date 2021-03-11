# %%

import matplotlib
matplotlib.use('QT4Agg')
from tablpy import table
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from tablpy.extra_funcs import extrem
from time import sleep

DataPath = "Données\\simulation\\"


show = True
# Fit sur C(w, h)


CD = table(DataPath + 'C(w,h)')
CD.renameCols('w h c')
CD['c'] *= -1
CD.giveUnits({'w': 'mm', 'h': 'mm', 'c': 'fF'})
CD.changeUnits({'c': 'F'})

w, h, c = CD[['w', 'h', 'c']].T
c.flatten()

w = w[::15]
h = h[:15:]
w, h = np.meshgrid(w, h)
c = c.reshape(11, 15).T


def plan(X, a, b, c):
    x, y = X
    return x*a + y*b + c


popts, pcov = curve_fit(plan, CD[['w', 'h']].T, CD['c'])
if show:
    x, y = np.arange(*extrem(CD['w']), 0.05), np.arange(*extrem(CD['h']), 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(plan((np.ravel(X), np.ravel(Y)), *popts))
    Z = zs.reshape(X.shape)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    sleep(5)

    ax.scatter(w, h, c, c=c.flatten(), cmap=plt.get_cmap("inferno"), label='data')
    ax.set_xlabel('Largeur')
    ax.set_ylabel('Hateur')
    ax.set_zlabel('Shunt Capacitance')
    ax.legend()
    fig.show()


def CF(w, h):
    return plan((w,h), *popts)


# %% Fit sur Cs(w, h) ---------------------------------------------------------
"""
CsD = table(DataPath + 'Cs(w,h)')
CsD.renameCols('w h cs')
CsD['cs'] *= -1
CsD.giveUnits({'w': 'mm', 'h': 'mm', 'cs': 'nF'})

w, h, cs = CsD[['w', 'h', 'cs']].T
cs.flatten()

w = w[::15]
h = h[:15:]
w, h = np.meshgrid(w, h)
cs = cs.reshape(11, 15).T


def plan(X, a, b, c):
    x, y = X
    return x*a + y*b + c


popts, pcov = curve_fit(plan, CsD[['w', 'h']].T, CsD['ca'])


if show:
    x, y = np.arange(*extrem(CsD['w']), 0.05), np.arange(*extrem(CsD['h']), 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(plan((np.ravel(X), np.ravel(Y)), *popts))
    Z = zs.reshape(X.shape)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    sleep(5)

    ax.scatter(w, h, cs, c=cs.flatten(), cmap=plt.get_cmap("inferno"), label='data')
    ax.set_xlabel('Largeur')
    ax.set_ylabel('Hateur')
    ax.set_zlabel('Shunt Capacitance')
    ax.legend()
    fig.show()


def CsF(w, h):
    return plan((w, h), *popts)

# %% fit C(l) -----------------------------------------------------------------
import matplotlib.pyplot as plt
from tablpy import table

def lin(x, a, b):
    return a*x + b

DataPath = "Données\\simulation\\"

CD = table(DataPath + 'C(w,h)')
CD.renameCols('w h c')
CD['c'] *= -1
CD.giveUnits({'w': 'mm', 'h': 'mm', 'c': 'nF'})
CD.newCol('l', '2*h + w')

CD.fit(lin, 'l', 'c')
plt.show()
"""
# %% Fit sur L(w, h) ---------------------------------------------------------


LD = table(DataPath + 'L(h,w)')
LD.renameCols('h w L')
LD.giveUnits({'w': 'mm', 'h': 'mm', 'L': 'nH'})
LD.changeUnits({'L': 'H'})
w, h, L = LD[['w', 'h', 'L']].T


w = w[:11:]
h = h[::11]
w, h = np.meshgrid(w, h)
L = L.reshape(11, 15)


def quad(X, a, b, c, d, f, g):
    x, y = X
    return a*x**2 + c*x*y + d*x + f*y + g


popts, pcov = curve_fit(quad, LD[['w', 'h']].T, LD['L'])

show=True
if show:
    x, y = np.arange(*extrem(LD['w']), 0.05), np.arange(*extrem(LD['h']), 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(quad((np.ravel(X), np.ravel(Y)), *popts))
    Z = zs.reshape(X.shape)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sur = ax.plot_surface(X, Y, Z, label='fit')
    sur._facecolors2d = sur._facecolor3d
    sur._edgecolors2d = sur._edgecolor3d

    sleep(5)

    ax.scatter(w, h, L, c=L.flatten(), cmap=plt.get_cmap("inferno"), label='data')
    ax.set_xlabel('Largeur')
    ax.set_ylabel('Hateur')
    ax.set_zlabel('Shunt Capacitance')
    ax.legend()
    fig.show()


def LF(w, h):
    return plan((w, h), *popts)

# %%
