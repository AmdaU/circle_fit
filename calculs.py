import matplotlib
import numpy as np
import fitter as xd
from lmfit import Model
from importlib import reload
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from progressbar import ProgressBar
from tablpy.extra_funcs import extrem
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit

DataPath = "Données\\simulation\\"

plt.rcParams['figure.figsize'] = 10, 6
matplotlib.rcParams.update({'font.size': 18})


def fix_csv(name):
    with open(name) as f:
        ls = f.readlines()

    with open(name, "w") as f:
        ls[0] = ls[0]   .replace('","', '_placeholder_')\
                        .replace(",", " ")\
                        .replace("_placeholder_", '","')
        f.writelines(ls)


# %%

xd = reload(xd)
xd.showUncertain = False

fix_csv(DataPath + 'C2.csv')

B = xd.results(DataPath + 'S12(x)_fin_2')
C = B.splitBy('Sub_xoff_mm')

# %%
Bdat = copy(B.data)

# %%
xd = reload(xd)
xd.showUncertain = False
B = xd.results(DataPath +'S12')
B.data = Bdat

dt = B.Analyse('high_res_mm')

# %%
for c in C:
    c.normalize(show=True)

# %%

from tablpy import *
import matplotlib
matplotlib.use('webAgg')
#import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from tablpy.extra_funcs import extrem
from time import sleep

DataPath = "Données\\simulation\\"

CsD = table(DataPath + 'Cs(w,h)')
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

popts, pcov = curve_fit(plan, CsD[['w','h']].T, CsD['c'])

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

# %%


## %




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')




X = CsD[['w', 'h']].T

popts, pcov = curve_fit(plan, X, CsD['c'])
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array(plan((np.ravel(X), np.ravel(Y)), *popts))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

# %%

def lin(x, a, b):
    return a*(x) + b


popt, pcov = curve_fit(lin, w, -c)
wtho = np.linspace(0.5, 15)
plt.plot()
plt.plot(w, -c)


CsD = table(DataPath + 'Cs(w,h)')

CsD.renameCols('w h c')
CsD.giveUnits({'w': 'mm', 'c': 'nF', 'h': 'mm'})


def lin2(params, a, b, c):
    x, y = params
    return a*x+b*y+c

w = CsD[]

CsD.fit2(lin, 'w', 'c')


# %%

def Cs(w, h, xoff):


x,y,z = a.T

curve_fit(lin, x, z)


def f(w, h, xoff):
    z0, eff = 168.9, 2.57
    cs = Cs(w, h, xoff)
    l = L(w, h, xoff)
    c = C(w, h, xoff)
    omega = np.linspace(6, 9, 100)
    plt.plot(omega, omega*Cs*Zo)
    plt.plot(omega, np.tan(omega*np.sqrt(eEff)/c))

# %%
import numpy as np
from ressonant_freq import fr
import matplotlib.pyplot as plt

L = 5.76e-9
C = 201.8e-15
Cs = 125.6e-15
epsilon = 2.57
l = (3.65*2 + 1)*1e-3

plt.title('Valeur de D.Zöpfl')
print(fr(L, C, Cs, epsilon, l, show=True))


plt.title('Nos valeurs')

L = 4.49e-9
C = 201.8e-15
Cs = 215e-15
epsilon = 2.57
l = (3.65*2 + 1)*1e-3

print(fr(L, C, Cs, epsilon, l, show=True))

L= np.logspace(-12,-6)

frs = []
for i in L:
    frs.append(fr(i, C, Cs, epsilon, l))

plt.plot(L, frs)
plt.show()
