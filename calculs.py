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


# %% Correlation des différent paramètres

xd = reload(xd)
xd.showUncertain = False

fix_csv(DataPath + 'Cs(w,h).csv')

B = xd.results(DataPath + 'S12(x)_fin_2')
B.giveUnits({'Sub_xoff_mm': 'mm', "f": "Hz"})
dt = B.Analyse('Sub_xoff_mm')

C = B.splitBy('Sub_xoff_mm')

for i in range(10):
    print(i)
    C[i].normalize(show=True)

C[5].subLin()
C[5].isolateSpike()

C[5].circleFit(show=True)

C[5].plot('re', 'im')
C[5].isolateSpike()

# %%
Bdat = copy(B.data)

# %%
xd = reload(xd)
xd.showUncertain = False
B = xd.results(DataPath +'S12')
B.data = Bdat


# %%
for c in C:
    c.magFit(show=True)



C[1].magFit()

C[1].plot('re', 'im')

C[1].circleFit(show=True, x0=[0.5, 0, 1])

C[0].ContainsRes

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

# %%

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
