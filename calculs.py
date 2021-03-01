import matplotlib
import numpy as np
import fit_perso as xd
from lmfit import Model
from importlib import reload
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from progressbar import ProgressBar
from tablpy.extra_funcs import extrem
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit

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

fix_csv('S12(x)_fin_2.csv')

B = xd.results('S12(x)_fin_2')
C = B.splitBy('Sub_xoff_mm')

# %%
Bdat = copy(B.data)

# %%
xd = reload(xd)
xd.showUncertain = False
B = xd.results('S12')
B.data = Bdat
B.renameCols({'Sub_xoff_mm': 'Delta_x'})

dt = B.Analyse('Delta_x')

# %%

xoff, frs = [], []

for i in range(len(C)//2):
    try:
        print(i)

        C[i].subLin()

        C[i].isolateSpike()

        #C[i].plot('re', 'im')
        C[i].circleFit(show=True, reject=True)
    except:
        print("something went wrong")

# %%
