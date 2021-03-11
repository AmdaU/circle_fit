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

fix_csv(DataPath + 'L(w,h).csv')

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
from ressonant_freq import *
import matplotlib.pyplot as plt
import matplotlib
from parameters import LF

LF(1, 3.65)

L = 5.76e-9
C = 201.8e-15
Cs = 125.6e-15
epsilon = 2.57
l = (3.65*2 + 1)*1e-3

plt.title('Valeur de D.Zöpfl')
print(fr(L, C, Cs, epsilon, l))
plt.cla()

plt.title('Nos valeurs')

L = 4.49e-9
C = 201.8e-15
Cs = 215e-15
epsilon = 2.57
l = (3.65*2 + 1)*1e-3

print(fr(L, C, Cs, epsilon, l))

L= np.logspace(-12,-6)

frs = []
for i in np.linspace(2.5, 4.5, 5):
    frs.append(frp(1, i, Cs, epsilon))

matplotlib.use('webagg')

plt.plot(np.linspace(2.5, 4.5, 5), frs)
plt.show()
plt.cla()
