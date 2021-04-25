# %%

import numpy as np
#import matplotlib
from ressonant_freq import*
# import matplotlib.pyplot as plt
from parameters import LF, CsF, CF
import matplotlib.pyplot as plt
from tablpy import table

CsF(1.5, 3.65)

DataPath = "Données\\simulation\\"

Cs = 125e-15
epsilon = 2.57

w = np.linspace(0.5, 1.5, 10)
h = np.linspace(2.5, 4.5 ,10)

w, h = np.meshgrid(w, h)

cmap='inferno'

w.shape

Z = np.zeros(w.shape)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = frp(w[i,j], h[i,j], epsilon)

plt.xlabel('Largeur')
plt.ylabel('Hauteur')
plt.pcolormesh(w,h, Z, cmap=cmap)
plt.colorbar()
plt.savefig('color_freq_theo.png')
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(w, h, Z, cmap=cmap)
ax.set_title("Fréquence de résonance théorique")
ax.set_xlabel('Largeur')
ax.set_ylabel('Hauteur')
ax.set_zlabel('Fréquence de résonance')


plt.savefig('3D_freq_theo.png')

plt.show()

# %%
