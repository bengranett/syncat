""" Draw samples from a distribution. 

Ben Granett
"""

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, BarycentricInterpolator, Akima1DInterpolator, InterpolatedUnivariateSpline


class sample_dist(object):
    """ """

    interp_methods = {
        'pchip': PchipInterpolator,              # smooth but may be peaky
        'linear': interp1d,                      # blocky
        'bary': BarycentricInterpolator,         # doesn't work
        'akima': Akima1DInterpolator,            # blocky artefacts
        'spline': InterpolatedUnivariateSpline,  # doesn't work
    }

    def __init__(self, bin_edges, bin_count, method='pchip'):
        """ Sample a distribution specified by a 1D histogram.

        Constructs samples from the inverse of the cumulative distribution
        function. The cumulative distribution function is interpolated.

        The interpolation method may be 'linear' or 'pchip'.  PCHIP stands
        for Piecewise Cubic Hermite Interpolating Polynomial
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
        PCHIP results in a smooth continuous distribution but may have
        undesired peaks.  Linear will reproduce the blocky histogram.

        Parameters
        ----------
        bin_edges : ndarray (length M + 1)
            array with bin edges
        bin_count : ndarray (length M)
            array with counts in bins (arbitrary normalization)
        method : str
            interpolation algorithm to use. Can be 'pchip', 'linear' (default 'pchip')
        """
        assert(len(bin_count) + 1 == len(bin_edges))
        assert(len(bin_count) > 2)

        assert method in self.interp_methods

        interper = self.interp_methods[method]

        cumsum = np.zeros(len(bin_edges), dtype='d')
        cumsum[1:] = np.cumsum(bin_count)
        cumsum = cumsum * 1. / cumsum[-1]

        self.f = interper(cumsum, bin_edges)

    def plot(self):
        """ """
        import pylab
        pylab.figure()
        xx = np.linspace(0, 1, 1000)
        pylab.plot(xx, self.f(xx))
        pylab.show()

    def __call__(self, n):
        """
        Parameters
        ----------
        n : int
            number of samples to draw

        returns
        -------
        ndarray : samples (length n)
        """
        assert(n > 0)

        u = np.random.uniform(0, 1, n)

        samples = self.f(u)

        return samples


def test():
    """ """
    import pylab
    edges = np.linspace(0, 2, 21)
    xc = (edges[1:] + edges[:-1]) / 2.
    y = np.exp(-(xc - 1.15)**2/2./0.2**2) + 2*np.exp(-(xc - 0.65)**2/2./0.1**2)
    # y = np.ones(len(xc))
    y = y * 1. / y.sum()

    xx = np.linspace(xc.min(),xc.max(),100)

    pylab.plot(xc, y, "o", c='k')
    # pylab.plot(xx, np.interp(xx, xc, y), dashes=(4,1), c='k')
    pylab.plot(xc, y, ls='steps-mid')

    S = sample_dist(edges, y, method='pchip')
    z = S(1e6)
    # S.plot()
    print "count",len(z)

    h,e = np.histogram(z, np.linspace(0, 2, 1000))
    h = h*1./h.sum() * (len(h)) * 1. / len(xc)
    pylab.plot((e[1:]+e[:-1])/2., h)


    h,e = np.histogram(z, edges)
    h = h*1./h.sum() * (len(h)) * 1. / len(xc)
    pylab.plot((e[1:]+e[:-1])/2., h, ls='steps-mid')

    pylab.show()

if __name__=="__main__":
    test()

