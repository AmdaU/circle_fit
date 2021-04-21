# %% Importations

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
from extra_funcs import *


plt.rcParams['figure.figsize'] = 10, 6
matplotlib.rcParams.update({'font.size': 18})

# %% Correlation des différent paramètres

DataPath = "..\\mesures\\simulation\\"
xd = reload(xd)
xd.showUncertain = False

fix_csv(DataPath + 'Cs(w,h).csv')

B = xd.results(DataPath + 'S12(x)_fin')
print(B)
B.giveUnits({'Sub_xoff_mm': 'mm', "f": "Hz"})
dt = B.Analyse('Sub_xoff_mm')

# %%
C = B.splitBy('Sub_xoff_mm')
# %%
C[0].plot('re', 'im')
plt.show()
C[0].subLin(show=True)
C[0].plot('re', 'im')

# %%
B.data

# %%
xd = reload(xd)
xd.showUncertain = False
B = xd.results(DataPath +'S12')
B.data = Bdat


# %%

n=10

print(C[n])
C[n].magFit()

C[n].plot('re', 'im')

C[n].circleFit(show=True, x0=[0.5, 0, 1])

C[n].ContainsRes

# %%
from tablpy import table
from tablpy import defVal

Cx = table(DataPath + 'C(x)')


Cx.renameCols('h w x c')
Cx.giveUnits({'c': 'fF', 'x': 'mm', 'w': 'mm'})
Cx['c'] *= -1

def lin(x, a, b):
    return a*(x) + b


defVal({'l' : '11 0 mm'})

Cx.newCol('d', 'l-x-w/2')
def inv(x, a, b):
    return a/x**(1.5) + b


Cx.data.sort_values(by='d', inplace = True)

std = np.std(Cx['c'])
avg = np.average(Cx['c'])

Cx.data = Cx.data[Cx['c']< avg+std*0.05]

Cx.plot('d', 'c')
plt.plot()

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

CsD.fit(lin, 'w', 'c')

# %% Fréquence de résonance théorique -----------------------------------------

a = table(DataPath + 'C(w,h)')
a.renameCols('w h c')

a.newCol('ct', 'frp(w, h)')
plt.cla()

# %%
L = 5.76e-9
C = 201.8e-15
Cs = 125.6e-15
epsilon = 2.57
l = (3.65*2 + 1)*1e-3

plt.title('Valeur de D.Zöpfl')
print(fr(L, C, Cs, epsilon, l))

# %%
from fitter import results
from tablpy import table

a = results(DataPath + 'S12')
print(a)
a.plot('f', '\\theta')
# %%
import tably
# %%
