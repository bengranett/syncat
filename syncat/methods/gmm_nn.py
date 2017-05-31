""" synrcat

gaussian mixture model

"""
import sys
import os
import numpy as np
import logging
from collections import OrderedDict

from astropy.table import Table
from sklearn.neighbors import KDTree

from pypeline import pype, add_param, depends_on

from syn import Syn
from gmm import GaussianMixtureModel
import syncat.misc as misc
import syncat.fileio as fileio

import time


@add_param('cat_model', metavar='filename', default='out/syn.pickle', type=str,
                        help='file with catalogue model to load')
@add_param('hints_file', metavar='filename', default='in/syn_hints.txt', type=str,
                        help='give hints about parameter distributions')
@add_param('cond_cat', metavar='filename', default='cond.fits', type=str, help='catalogue with given values')
@add_param('cond_cat_format', metavar='fmt', default='fits', type=str, help='astropy.table format code of catalogue file')
@add_param('cond_cat_columns', metavar='a', default=None, type=str, nargs='*', help='column names for ascii tables without header')
@add_param('cond_params', metavar='name', default=('z',), nargs='*', type=str,
                        help='list of given parameters')
@add_param('nn_sampling_factor', metavar='f', default=2, type=float, help='nn sample size')
@add_param('output_extra_columns', metavar='b', default=True, type='bool', help='output match distance in table')
@depends_on(Syn)
class NNMixtureModel(GaussianMixtureModel):
    """ SynCat mode to generate random catalogue by sampling from a gaussian mixture model.

    Parameters
    ----------
    mask : minimask.Mask instance
        mask describing survey geometry to sample from.  If None, sample from full-sky.
    cat_model : str
        path to file with catalogue model to load
    hints_file : str
        path to file with hints about parameter distributions
    """

    def __init__(self, config={}, mask=None, **kwargs):
        """ """
        self._parse_config(config, **kwargs)
        self._setup_logging()

        self.load_hints()

        self.mask = mask

        self.syn = None

    def sample(self):
        """ Sample from the Gaussian mixture model.

        Returns
        -------
        numpy strucarray : random catalogue
        """

        cond_table = fileio.read_catalogue(self.config['cond_cat'], format=self.config['cond_cat_format'], columns=self.config['cond_cat_columns'])

        dtype = self.syn.dtype

        dtype = misc.append_dtypes(self.syn.dtype, self.config['add_columns'], cond_table.dtype)

        if self.config['output_extra_columns']:
            dtype = misc.append_dtypes(dtype, self.config['cond_params'], cond_table.dtype, translate=lambda x: '_'+x)
            dtype = misc.concatenate_dtypes([dtype, np.dtype([('_dist_nn', 'f')])])

        out = np.zeros(len(cond_table), dtype=dtype)

        cond_data = misc.flatten_struc_array(cond_table[self.config['cond_params']].as_array())

        nrandom = int(len(cond_data)*self.config['nn_sampling_factor'])

        self.logger.debug("N random %i", nrandom)

        randoms = self.syn.sample(n=nrandom)

        rand_data = misc.struc_array_columns(randoms, self.config['cond_params'])

        self.logger.debug("Planting neighbor lookup tree ... (n=%i, dim=%i)", *rand_data.shape)

        mu = rand_data.mean(axis=0)
        sig = rand_data.std(axis=0)

        self.logger.debug("means: %s", mu)
        self.logger.debug("sigmas: %s", sig)

        rand_data_t = (rand_data - mu) / sig
        cond_data_t = (cond_data - mu) / sig

        tree = KDTree(rand_data_t)

        self.logger.debug("Querying nearest neighbors ... (n=%i, dim=%i)", *cond_data.shape)

        distance, matches = tree.query(cond_data_t, k=1, dualtree=True, return_distance=True)
        matches = matches[:, 0]
        distance = distance[:, 0]
        self.logger.debug("Match stats - max distance: %f, mean: %f", np.max(distance), np.mean(distance))

        if self.config['output_extra_columns']:
            out['_dist_nn'] = distance

        randoms = randoms[matches]

        columns_added = misc.insert_column(cond_table.columns, cond_table, out)
        columns_added = misc.insert_column(randoms.dtype.names, randoms, out, columns_added=columns_added)

        if self.config['output_extra_columns']:
            misc.insert_column(config['cond_params'], randoms, out, columns_added=columns_added, translate=lambda x: '_'+x)

        self.logger.debug("Output columns: %s", columns_added)

        return out
