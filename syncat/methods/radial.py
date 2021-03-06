""" synrcat

radial selection function

"""
import sys
import os
import numpy as np
from collections import OrderedDict

from astropy.table import Table

from pypeline import pype, add_param

import syncat.misc as misc
from syncat.errors import NoPoints

from sample_dist import sample_dist

import time


@add_param('zdist_file', metavar='filename', default='in/zdist.txt', type=str,
    help='path to file specifying redshift distribution')
@add_param('zdist_interp', metavar='name', default='pchip', choices=('linear','pchip'), type=str,
    help='method to interpolate cumulative distribution function')
class Radial(pype):
    """ SynCat mode to generate a random catalogue by drawing redshift from a distribution.

    Parameters
    ----------
    mask : minimask.Mask instance
        mask describing survey geometry to sample from.  If None, sample from full-sky.
    zdist_file : str
        path to file with catalogue model to load
    zdist_interp : str
        method to interpolate cumulative distribution function (default pchip)
    zdist : tuple
        redshift distribution histogram in tuple form (bin_edges, n_z)
    """

    def __init__(self, config={}, mask=None, zdist=None, **kwargs):
        """ """
        self._parse_config(config, **kwargs)
        self._setup_logging()

        self.zdist = zdist
        if zdist is None:
            self.load_zdist()

        self.mask = mask

    def load_zdist(self, filename=None):
        """ Load a redshift distribution from a file.

        The file format can be ascii with columns redshift and n(z)
        or a tuple (z, nz) stored in a numpy .npy or pickle file.

        Parameters
        ----------
        filename : str
            Path to redshift distribution file.
        """
        if filename is None:
            filename = self.config['zdist_file']

        if not os.path.exists(filename):
            self.logger.warning("redshift distribution file %s does not exist", filename)
            return

        try:
            zz, nz = np.load(filename)
            done = True
        except:
            pass

        try:
            zz, nz = np.loadtxt(filename, unpack=True)
            done = True
        except:
            pass

        if not done:
            self.logger.warning("Could not load file %s.  Unknown format.", filename)
            return

        # normalize redshift distribution
        nz = nz * 1. / np.sum(nz)

        step = zz[1] - zz[0]
        bin_edges = np.arange(len(zz) + 1) * step + zz[0] - step / 2.

        self.zdist = (bin_edges, nz)
        self.logger.info("Loaded redshift distribution file %s.", filename)

    def sample_sky(self, zone=None, nside=None, order=None):
        """ Sample sky coordinates.

        Parameters
        ----------
        zone : int, list
            optional healpix zone index or list of indices from which to sample.  Otherwise sample from all zones.
        nside : int
            healpix nside for zone pixelization
        order : str
            healpix ordering for zone pixelization
        """
        return np.transpose(self.mask.draw_random_position(density=self.config['density'], n=self.config['count'],
                                                            cell=zone, nside=nside, order=order))

    def fit(self):
        """ """
        pass

    def sample(self, add_columns=(), zone=None, nside=None, order=None):
        """ Draw samples from the radial selection function.

        Returns
        -------
        numpy strucarray : random catalogue
        """
        if self.config['overwrite']:
            if os.path.exists(self.config['out_cat']):
                self.logger.info("overwriting existing catalogue: %s", self.config['out_cat'])
                os.unlink(self.config['out_cat'])

        bin_edges, bin_counts = self.zdist

        sampler = sample_dist(bin_edges, bin_counts)

        skycoord = self.sample_sky(zone=zone, nside=nside, order=order)
        if len(skycoord) == 0:
    		raise NoPoints

        redshift = sampler(len(skycoord))

        data_out = OrderedDict({'z': redshift})

        skycoord_name = self.config['skycoord_name']
        for i in range(len(skycoord_name)):
            data_out[skycoord_name[i]] = skycoord[:, i]

        for column, t in add_columns:
            data_out[column] = np.zeros(len(redshift), dtype=t)

        data_out = misc.dict_to_structured_array(data_out)

        self.logger.info("Wrote radial random catalogue nobj=%i: %s", len(data_out), self.config['out_cat'])

        return data_out
