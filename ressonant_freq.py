import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


def fr(L, C, Cs, epsilon, l, show=False, n=1):
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
            plt.scatter(sol, f2(sol), color='r')

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
