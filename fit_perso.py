import warnings
import numpy as np
from lmfit import Model
from tablpy import table
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from scipy.stats import pearsonr
from tablpy.extra_funcs import roll, extrem

table.showUncertain = False


def lin(x, a, b):
    return a*x + b


def quad(x, a, b, c):
    return a*x**2 + b*x + c


class results(table):
    """
    results object based on tablpy's talbe

    https://github.com/AmdaUwU/tablpy

    It is assumed that the last columns of the table are :Frequency (GHz),
    re(S_12), im(S_12), in this order.
    If that is not the case, you can change the names manually with the
    .renameCols method and the units with the .giveUnits mehtod

    """
    __doc__ += table.__doc__

    def __init__(self, fichier, db=False, data=None, AutoInsert=True, **kargs):
        super().__init__(fichier, AutoInsert=AutoInsert, data=data)


        #
        if data is None:
            noms = list(self.data)[:-6:2]
            self.renameCols(' '.join(noms + ['f re im']))
            self.giveUnits({'f': 'GHz'})
            self.newCol(r'\theta', 'atan(im/re)', units='rad')
            self.newCol(r'mag', 'sqrt(re**2 + im**2)', units='1')
            for i in kargs:
                exec(f"self.{i} = kargs[i]")
        self.IsSubLined = False
        self.IsIsolated = False
        self.IsRejected = False
        self.IsNormalized = False
        self.ContainsRes = True
        self.failed = False

    def updateCart(s):
        """Ensure cartesian coordinates are consistant with the polar ones"""
        s['re'], s['im'] = (a(s['\\theta'])*s['mag'] for a in (np.cos, np.sin))

    def updatePolar(self):
        """Ensure polar coordinates are consistant with the cartesian ones"""
        self['\\theta'] = np.arctan2(*self[['im', 're']].T)
        self['mag'] = np.sqrt(self['re']**2+self['im']**2)

    def subLin(self):
        """ removes linear background from data """
        popt, pcov = self.fit(lin, 'f', '\\theta', show=False)
        self['\\theta'] = self['\\theta'] - lin(self['f'], *popt)
        #self['\\theta'] = roll(self['\\theta'], -np.pi/2, np.pi/2)
        self.updateCart()
        self.IsSubLined = True

    def circleFit(self, x0=None, show=False, reject=True):
        '''
        return center_x, center_y, radius of circle

        (x0) guess for circle Parameters

        (show) plots data and fit, mainly for debuging

        (reject) If true, reject points laying too far from the fitted circle
        '''
        # genere une copie sur les quels les transformations seront fait
        t = copy(self)
        # deplace l'origine a l'interieur du cercle
        xoff, yoff = np.average(t['re']), np.average(t['im'])
        t['re'] -= xoff
        t['im'] -= yoff
        t.updatePolar()

        if x0 is None:
            x0 = [(max(t['re'])+min(t['re']))/2, (max(t['im'])+min(t['im']))/2,
                  (max(t['im'])-min(t['im']))/2]

        # Defines a circle equation in polar coordinates
        def circle(theta, a, b, r):
            return a*np.cos(theta) + b*np.sin(theta)\
                   + np.sqrt((a*np.cos(theta)+b*np.sin(theta))**2-a*a-b*b+r*r)

        # first fit!
        Circle = Model(circle)
        params = Circle.make_params(a=x0[0], b=x0[1], r=x0[2])

        results = Circle.fit(t['mag'], params, theta=t['\\theta'])
        vals = results.values.values()

        # rejection of outliers points
        diff = t['mag']-circle(t['\\theta'], *vals)
        std = np.std(diff)

        # arbitrairy amount of standard deviation allowed
        n = 1

        if reject and not self.IsRejected:
            t.data = t.data[diff < n*std]
            self.data = self.data[diff < n*std]
            self.IsRejected = True

        # second fit without rejected points
        results = Circle.fit(t['mag'], params, theta=t['\\theta'])
        vals2 = results.values.values()
        theta = np.linspace(-np.pi, np.pi, 1000)

        if show:
            re, im = (a(theta)*circle(theta, *vals2) for a in (np.cos, np.sin))
            plt.plot(re, im, label='fit')
            rem = np.cos(theta)*(circle(theta, *vals) + n*std)
            imm = np.sin(theta)*(circle(theta, *vals) + n*std)
            plt.plot(rem, imm, label='max deviation')

            t.plot('re', 'im')
            plt.gca().set_aspect(1)
            plt.legend()
            plt.show()
        return np.array(list(vals2)) + np.array([xoff, yoff, 0])

    def normalize(self, x0=None, x0c=None, show=False, isolate=True):
        """Modifiy the data by normalizing it"""
        # Ensures linear backgrond is removed and the spike is IsIsolated
        if not self.IsSubLined:
            self.subLin()
            self.IsSubLined = True
        if not self.IsIsolated and isolate:
            self.isolateSpike()
            self.IsIsolated = True
        if not self.ContainsRes:
            warnings.warn("The resonant Frequency appeares not to be in the "
                          "frequency range")
            return None
        if show:
            plt.title('Determining center and radius')
        a, b, r = self.circleFit(x0=x0c, show=show)

        t = copy(self)

        # Centering the circle
        t['re'] -= a
        t['im'] -= b

        # Guessing initials parameters
        if x0 is None:
            self.guess_theta_0 = np.average(t['\\theta'])
            self.guess_Q = self.guess_freq_res/self.guessWidth*5
            x0 = [self.guess_theta_0, self.guess_Q, self.guess_freq_res]

        # def theta(x, theta0, Ql, fr):
            # return roll((theta0 + 2*np.arctan(2*Ql*(1-x/fr))), -np.pi/2, np.pi/2)

        def theta2(x, theta0, Ql, fr):
            return roll((theta0 + np.arctan(2*Ql*(1-x/fr)))-np.pi/2, -np.pi/2, np.pi/2)

        #popt, pcov = curve_fit(theta2, *t[['f', '\\theta']].T, p0=x0)
                               #sigma=t['\\theta']/t['mag'])
        Theta = Model(theta2)
        params = Theta.make_params(
            theta0=self.guess_theta_0,
            Ql=self.guess_freq_res/self.guessWidth*5,
            fr=self.guess_freq_res)
        freq = self['f']

        weight = (10/(0.5+((self['f'] - self.guess_freq_res)/self.guessWidth)**2))

        #weight=1
        result = Theta.fit(self['\\theta'], params, x=freq, weights=weight, max_nfev=10000)
        self.guess_freq_res = result.values['fr']
        self.guess_Ql = result.values['Ql']
        x = a + r*np.cos(result.values['theta0'])
        y = b + r*np.sin(result.values['theta0'])
        if show:
            t.plot("f", "\\theta")
            f = np.linspace(*extrem(t['f']), 1000)
            #plt.plot(f, theta2(f, *popt), label="fit")
            plt.plot(f, theta2(f, *result.values.values()), label='fit2')
            plt.plot(f, theta2(f, *params.valuesdict().values()), label='guess')
            plt.legend()
            plt.show()
            self.plot('re', 'im')
            plt.gca().set_aspect(1)
            plt.scatter(x, y, s=200, c='orange', label='off-resonant point')
            plt.legend()
            plt.gca().spines['bottom'].set_position('zero')
            plt.show()
        self['mag'] /= np.sqrt(x*x + y*y)
        self['\\theta'] -= np.arctan2(y, x)
        self.updateCart()
        self.IsNormalized = True

    def splitBy(self, col):
        out = []
        for x in list(set(self[col])):
            sub_df = self.data.loc[self[col] == x]
            exec(r'r=results("", data=sub_df, {}={}, AutoInsert = False)'
                 .format(col, x), locals(), globals())
            exec(f'r.{col} = {x}')
            r.data.index = (np.arange(0, len(r)))
            out.append(r)
            shit = f"out[-1].delCol('{col}')"
            exec(shit)
        return out

    def isolateSpike(self, show=False):
        avg = np.average(self['mag'])
        std = np.std(self['mag'])
        minmag = float(self.data[self['mag'] == min(self['mag'])]['f'])
        maxmag = float(self.data[self['mag'] == max(self['mag'])]['f'])
        self.guess_freq_res =\
            float(self.data[self['\\theta'] == min(self['\\theta'])]['f'])
        low_freq = self.data.loc[self['f'] < minmag]
        high_freq = self.data.loc[self['f'] > maxmag]
        if len(low_freq) == 0 or len(high_freq) == 0:
            self.ContainsRes = False
            return None
        n = 2
        if show:
            plt.hlines(avg - n*std, *extrem(self['f']))
            plt.hlines(avg + n*std, *extrem(self['f']))
            plt.vlines(minmag, 0, 1)
        keep = low_freq[abs(low_freq['mag'] - avg) < n*std]
        keep2 = high_freq[abs(high_freq['mag'] - avg) < n*std]
        x0 = keep.index[-1]
        x1 = keep2.index[0]
        if show:
            plt.plot(*self[['f', 'mag']][keep.index].T, ".")
            plt.plot(*self[['f', 'mag']][keep2.index].T, ".")
            plt.vlines(self['f'][x0], 0, 0.5, linestyles='dashed', color='g')
            plt.vlines(self['f'][x1], 0, 0.5, linestyles='dashed', color='g')
            plt.show()
        index_width = x1-x0
        self.guessWidth = self['f'][x1] - self['f'][x0]
        self.data = self.data[x0-index_width: x1+index_width]
        self.IsIsolated = True

    def magFit(self, show=False):
        if not self.IsNormalized:
            self.normalize()
        x0 = [self.guess_freq_res,
              self.guess_freq_res/self.guessWidth*10,
              self.guess_freq_res/self.guessWidth*10,
              np.pi/8,
              -0.1
              ]
        Mod_S12 = Model(S12_fit)
        params = Mod_S12.make_params(
            fr=self.guess_freq_res,
            Ql=self.guess_Ql,
            Qc=self.guess_Ql*2,
            phi=0,
            offset=0)

        weight = (100/(1+((self['f'] - self.guess_freq_res)/self.guessWidth)**2))

        result = Mod_S12.fit(self['mag'], params, x=self['f'], weights=weight)
        p = result.params
        if show:
            result.plot()
        if not result.errorbars\
           or np.any([abs(p[a].stderr/p[a].value) > 1 for a in p.keys()]):
            self.failed = True

        self.freq_res = result.values['fr']
        self.Ql = result.values['Ql']
        self.QC = result.values['Qc']
        self.phi = result.values['phi']
        self.offset = result.values['offset']
        return result

    def Analyse(self, var, params=['fr', 'Ql', 'Qc', 'phi'], **kwargs):
        SubDts = self.splitBy(var)
        data = []
        for sub in SubDts:
            results = sub.magFit().params
            if not sub.failed:
                exec(f"slice = [sub.{var}, 0]", locals(), globals())
                for i in params:
                    slice.extend([results[i].value, results[i].stderr])
                data.append(slice)
        dt = table("", data=data, AutoInsert=False)
        dt.renameCols([var] + params)
        dt.giveUnits({'fr': sub.units['f']})
        for i in params:
            dt.plot(var, i)
            corr, p = pearsonr(*dt[[var, i]].T)
            plt.annotate('r = {:.3f}\np = {:.3g}'.format(corr, p),
                         xy=(0.05, 0.25), xycoords='axes fraction')
            plt.show()
        return dt

def S12_fit(x, fr, Ql, Qc, phi, offset):
    return np.abs(1-Ql/Qc*np.exp(-1j*phi)/(1+2j*Ql*(x-fr)/fr)) + offset