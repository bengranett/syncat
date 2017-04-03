""" synrcat

shuffle mode

"""
import sys
import os
import numpy as np

from astropy.table import Table

from pypeline import pype

import pypelid.utils.misc as misc

import time


class Shuffle(pype):
    """ SynCat mode to generate a random catalogue by shuffling."""
    _default_params = {}

    def __init__(self, config={}, mask=None, **kwargs):
        """ """
        self._parse_config(config, **kwargs)
        self._setup_logging()
        self.mask = mask

    def sample_sky(self, zone=None, nside=None, order=None):
        """ """
        return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=self.config['count'],
                                                            cell=zone, nside=nside))

    def fit(self):
        """ """
        pass

    def sample(self, filename=None):
        """ shuffle catalogue. """

        if filename is None:
            filename = self.config['in_cat']

        self.logger.info("loading %s", filename)

        if self.config['overwrite']:
            if os.path.exists(self.config['out_cat']):
                self.logger.info("overwriting existing catalogue: %s", self.config['out_cat'])
                os.unlink(self.config['out_cat'])

        # load full catalogue to shuffle
        data = Table.read(filename)

        skycoord = self.sample_sky()

        data_out = np.random.choice(data, size=len(skycoord), replace=True)

        skycoord_name = misc.ensurelist(self.config['skycoord_name'])
        dim = len(skycoord_name)
        if dim == 1:
            data_out[skycoord_name[0]] = skycoord
        else:
            for i in range(dim):
                data_out[skycoord_name[i]] = skycoord[:, i]

        self.logger.info("Wrote shuffled catalogue nobj=%i: %s", len(data_out), self.config['out_cat'])

        return data_out
