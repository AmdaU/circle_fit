# %%

import numpy as np
from scipy.optimize import brentq
from parameters import CF, LF, CsF

def fr(L, C, Cs, epsilon, l, show=False, n=1):
    """
    returns de theoretical frequency of the resonator

    L :      Inductance
    C :      Capacitance
    Cs:      Capacitance Shunt
    epsilon: permitivity if the substrate
    l :      total lenght
    """
    c = 299792485
    r = np.sqrt(epsilon)/c*l
    Z0 = np.sqrt(L/C)

    def f1(w):
        return -Cs*Z0*w*1e9

    def f2(w):
        return np.tan(w*1e9*r)

    smol = 0.00000000001
    sols = []
    for i in range(n):
        x = (i + 1/2)*(np.pi)/(1e9*r)+smol
        x2 = (i + 1 + 1/2)*(np.pi)/(1e9*r)-smol
        sol = brentq(lambda x: f1(x) - f2(x), x, x2)
        sols.append(sol/(2*np.pi))
        if show:
            plt.scatter(sol, f2(sol), s =2, color='r')

    if show:
        w = np.linspace(0, n*np.pi/(1e9*r), 100000)
        plt.plot(w, f1(w), label=r'$-\omega C_s Z_0$')
        plt.plot(w, f2(w), ',', label=r'tan$\beta l$')
        plt.ylim(f1(w[-1])-1, 1)
        plt.legend()
        plt.show()
    if n == 1:
        return sols[0]
    return sols

def frp(w, h, epsilon):
    """
    Return the renonant frequency based on width and hight of the 
    resonater instead of L, C, Cs and epsilon
    """
    C = CF(w, h)
    L = LF(w, h)
    Cs = CsF(w, h)
    return fr(L, C, Cs, epsilon, (2*h + w)*1e-3)
