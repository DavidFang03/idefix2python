import numpy as np


def fit(X, Y, deg, start=0, end=1):
    index_start = int(start * (len(X) - 1))
    index_end = int(end * (len(X) - 1))
    params, cov = np.polyfit(
        X[index_start:index_end], Y[index_start:index_end], deg=deg, cov=True
    )
    return params, np.diag(cov)


def integrate(df, f0, times):
    values = [f0]
    dt = np.diff(times)
    for i, t in enumerate(times[:-1]):
        previousValue = values[-1]
        value = previousValue + df(previousValue) * dt[i]
        values += [value]
    return values


def vK(r):
    return r ** (-0.5)


class Fluid:
    def __init__(self, cs0, csSlope, sigma0, sigmaSlope, Stokes0):
        self.cs0 = cs0
        self.csSlope = csSlope
        self.sigma0 = sigma0
        self.sigmaSlope = sigmaSlope
        self.rhoSlope = sigmaSlope - 1
        # # HSlope = csSlope + 0.5
        # # self.rhoSlope = sigmaSlope + csSlope - 0.5
        # self.rhoSlope = sigmaSlope
        # self.cs0 = 0.05
        # self.csSlope = -0.5
        # self.rhoSlope = -1.5
        self.Stokes0 = Stokes0

    def eta(self, r):
        cs0 = self.cs0
        csSlope = self.csSlope
        rhoSlope = self.rhoSlope
        cs2 = cs0**2 * np.pow(r, 2 * csSlope)
        return cs2 / (vK(r) ** 2) * (2 * csSlope + rhoSlope)

    def Stokes(self, r):
        Stokes0 = self.Stokes0

        # return Stokes0
        OmegaK = r ** (-1.5)
        return Stokes0 * OmegaK
        # return Stokes0 * self.sigma0 * OmegaK / r ** (self.csSlope)

    def vrDrift(self, r):
        st = self.Stokes(r)
        return self.eta(r) * vK(r) / (st + 1 / st)

    # def eta(self, r):
    #     cs0 = self.cs0
    #     csSlope = self.csSlope
    #     rhoSlope = self.rhoSlope
    #     return -(rhoSlope + 2 * csSlope) * cs0 * cs0 * r ** (2 * csSlope + 1)

    # def Stokes(self, r):
    #     Stokes0 = self.Stokes0
    #     rhoSlope = self.rhoSlope
    #     csSlope = self.csSlope
    #     return Stokes0 * r ** (-1.5 - rhoSlope - csSlope)

    # def vrDrift(self, r):
    #     return -self.eta(r) / np.sqrt(r) / (self.Stokes(r) + 1 / self.Stokes(r))
