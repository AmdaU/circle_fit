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

fix_csv(DataPath + 'L(w,h).csv')

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

# %% Fréquence de résonance théorique
import numpy as np
#import matplotlib
from ressonant_freq import*
# import matplotlib.pyplot as plt
from parameters import LF
from tablpy import table
import matplotlib.pyplot as plt

DataPath = "Données\\simulation\\"

Cs = 125e-15
epsilon = 2.57

w = np.linspace(0.5, 1.5, 10)
h = np.linspace(2.5, 4.5 ,10)

w, h = np.meshgrid(w, h)

w.shape

Z = np.zeros(w.shape)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = frp(w[i,j], h[i,j], Cs, epsilon)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(w, h, Z)
ax.set_title("Fréquence de résonance théorique, Capacitance Shunt négligé")
ax.set_xlabel('Largeur')
ax.set_ylabel('Hateur')
ax.set_zlabel('Fréquence de résonance')

plt.show()
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
