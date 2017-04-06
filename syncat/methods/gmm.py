""" synrcat

gaussian mixture model

"""
import sys
import os
import numpy as np
from collections import OrderedDict

from astropy.table import Table

from pypeline import pype, add_param, depends_on

from syn import Syn

import time


@add_param('cat_model', metavar='filename', default='out/syn.pickle', type=str,
                        help='file with catalogue model to load')
@add_param('hints_file', metavar='filename', default='in/syn_hints.txt', type=str,
                        help='give hints about parameter distributions')
@depends_on(Syn)
class GaussianMixtureModel(pype):
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
        return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=self.config['count'],
                                                            cell=zone, nside=nside))

    def load_hints(self):
        """ Load the hints file.
        The hints file contains information about the parameter distributions.
        """
        self.hints = {}

        if os.path.exists(self.config['hints_file']):
            for line in file(self.config['hints_file']):
                line = line.strip()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue

                words = line.split()

                instruction = None
                low = None
                high = None

                name = words.pop(0)

                if len(words) > 0:
                    instruction = words.pop(0)

                if len(words) > 0:
                    low = float(words.pop(0))

                if len(words) > 0:
                    high = float(words.pop(0))

                if instruction not in self.hints:
                    self.hints[instruction] = []
                self.hints[instruction].append((name, low, high))

                self.logger.info("got hint for '%s': instruction is %s with range: %s, %s", name, instruction, low, high)

        return self.hints

    def fit(self, filename=None):
        """ Fit a Gaussian mixture model to the input catalogue.

        Parameters
        ----------
        filename : str
            path to input catalogue.
        """
        if filename is None:
            filename = self.config['in_cat']

        if os.path.exists(self.config['cat_model']) and not self.config['overwrite']:
            self.logger.info("reading %s", self.config['cat_model'])
            self.syn = Syn(self.config['cat_model'])
            self.labels = self.syn.labels
            return

        hints = self.load_hints()

        self.logger.info("loading %s", filename)

        table = Table.read(filename, format=self.config['input_format'])

        other_dtypes = {}
        properties = []

        for name in table.columns:
            hit = False
            for skip in self.config['skip']:
                if skip.lower() in name.lower():
                    hit = True
                    self.logger.info("ignoring column '%s' because it includes the string '%s'.", name, skip)
                    other_dtypes[name] = np.dtype([(name.encode('ascii'), table.dtype[name])])
                    break

            if not hit:
                properties.append(name)

        table = table[properties]

        if self.logger.isEnabledFor(logging.INFO):
            mesg = ""
            for i, p in enumerate(properties):
                mesg += "\n{:>3} {}".format(1 + i, p)
            self.logger.info("got these %i columns:%s", len(properties), mesg)

        self.syn = Syn(labels=properties, hints=hints, config=self.config)

        dtype = table.dtype
        for name in self.config['add_columns']:
            try:
                dtype = misc.concatenate_dtypes([dtype, other_dtypes[name]])
            except KeyError:
                pass

        if self.config['sample_sky'] and self.config['skycoord_name'] not in dtype.names:
            skycoord_name = misc.ensurelist(self.config['skycoord_name'])
            dim = len(skycoord_name)

            if dim == 1:
                skycoord_dtype = np.dtype([(skycoord_name[0], np.dtype((np.float64, 2)))])
            elif dim == 2:
                alpha, delta = skycoord_name
                skycoord_dtype = np.dtype([(alpha, np.float64), (delta, np.float64)])

            dtype = misc.concatenate_dtypes([dtype, skycoord_dtype])

        print dtype

        if self.config['quick']:
            table = table[:10000]

        self.syn.fit(table, dtype=dtype)

        # store column names
        self.labels = properties

        # save catalogue model
        self.syn.save(self.config['cat_model'])

    def sample(self):
        """ Sample from the Gaussian mixture model.

        Returns
        -------
        numpy strucarray : random catalogue
        """
        if self.syn is None:
            if not os.path.exists(self.config['cat_model']):
                raise Exception("Cannot load catalogue model.  Files does not exist: %s"%self.config['cat_model'])
            self.syn = Syn(self.config['cat_model'])

        if self.config['sample_sky']:
            skycoord = self.sample_sky()
            count = len(skycoord)
        else:
            count = self.config['count']

        randoms = self.syn.sample(n=count)
        if self.config['sample_sky']:
            skycoord_name = misc.ensurelist(self.config['skycoord_name'])

            dim = len(skycoord_name)

            if dim == 1:
                randoms[skycoord_name[0]] = skycoord
            else:
                for i in range(dim):
                    randoms[skycoord_name[i]] = skycoord[:,i]

        return randoms
