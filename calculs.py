# Ce fichier utilise fitter.py pour traiter les données de HFSS

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

B = xd.results(DataPath + 'S12(x)_fin.csv')
B.renameCols({'Sub_xoff [mm]': 'dx'})
B.giveUnits({'dx': 'mm', "f": "GHz"})
dt = B.Analyse('dx', save=True)

# %%
B.data

# %%
C = B.splitBy('dx')
for i in range(len(C)):
    plt.title(i)
    C[0].magFit(show=True)



# %% Fréquence de résonance théorique -----------------------------------------

a = table(DataPath + 'C(w,h)')
a.renameCols('w h c')

a.newCol('ct', 'frp(w, h)')
plt.cla()
