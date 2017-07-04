""" synrcat

shuffle mode

"""
import sys
import os
import numpy as np

from astropy.table import Table

from pypeline import pype

import syncat.fileio as fileio
import syncat.misc as misc
from syncat.errors import NoPoints

import time


class Shuffle(pype):
    """ SynCat mode to generate a random catalogue by shuffling.

    Parameters
    ----------
    mask : minimask.Mask instance
        mask describing survey geometry to sample from.  If None, sample from full-sky.
    """

    _default_params = {}

    def __init__(self, config={}, mask=None, **kwargs):
        """ """
        self._parse_config(config, **kwargs)
        self._setup_logging()
        self.mask = mask

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
                                                            cell=zone, nside=nside))

    def fit(self):
        """ """
        pass

    def sample(self, filename=None):
        """ shuffle catalogue. 

        Parameters
        ----------
        filename : str
            optional path to input catalogue

        Returns
        -------
        numpy strucarray : random catalogue
        """

        if filename is None:
            filename = self.config['in_cat']

        self.logger.info("loading %s", filename)

        if self.config['overwrite']:
            if os.path.exists(self.config['out_cat']):
                self.logger.info("overwriting existing catalogue: %s", self.config['out_cat'])
                os.unlink(self.config['out_cat'])

        skycoord = self.sample_sky()

        if len(skycoord) == 0:
            raise NoPoints

        # load full catalogue to shuffle
        table = fileio.read_catalogue(filename, format=self.config['input_format'], columns=self.config['input_columns'])

        table = misc.remove_columns(table, self.config['skip'])
        dtype = table.dtype

        skycoord_name = self.config['skycoord_name']
        skycoord_dim = len(skycoord_name)

        if skycoord_dim == 1:
            skycoord_dtype = np.dtype([(skycoord_name[0], np.dtype((np.float64, 2)))])
        elif skycoord_dim == 2:
            alpha, delta = skycoord_name
            skycoord_dtype = np.dtype([(alpha, np.float64), (delta, np.float64)])
        else:
            raise ValueError("skycoord_name must be length 1 or 2, not %s"%skycoord_dim)

        try:
            dtype = misc.concatenate_dtypes([dtype, skycoord_dtype])
        except ValueError:
            pass

        data_out = np.zeros(len(skycoord), dtype=dtype)

        shuffle = np.random.choice(table, size=len(skycoord), replace=True)

        for name in shuffle.dtype.names:
            data_out[name] = shuffle[name]


        if skycoord_dim == 1:
            data_out[skycoord_name[0]] = skycoord
        else:
            for i in range(skycoord_dim):
                data_out[skycoord_name[i]] = skycoord[:, i]

        self.logger.info("Wrote shuffled catalogue nobj=%i: %s", len(data_out), self.config['out_cat'])

        return data_out
