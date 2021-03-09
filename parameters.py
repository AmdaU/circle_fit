from tablpy import *
import matplotlib
matplotlib.use('webAgg')
#import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from tablpy.extra_funcs import extrem
from time import sleep


# Fit sur C(w, h)

DataPath = "Données\\simulation\\"

CsD = table(DataPath + 'C(w,h)')
CsD.renameCols('w h c')
CsD['c'] *= -1
CsD.giveUnits({'w': 'mm', 'h': 'mm', 'c': 'nF'})

w, h, c = CsD[['w', 'h', 'c']].T
c.flatten()

w = w[::15]
h = h[:15:]
w, h = np.meshgrid(w, h)
c = c.reshape(11, 15).T


def plan(X, a, b, c):
    x, y = X
    return x*a + y*b + c


popts, pcov = curve_fit(plan, CsD[['w', 'h']].T, CsD['c'])

x, y = np.arange(*extrem(CsD['w']), 0.05), np.arange(*extrem(CsD['h']), 0.05)
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


def CsF(w, h):
    return plan((w,h), *popts)


# Fit sur Cs(w, h) ------------------------------------------------------------