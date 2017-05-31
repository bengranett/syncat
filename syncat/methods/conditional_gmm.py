""" synrcat

gaussian mixture model

"""
import sys
import os
import numpy as np
import logging
from collections import OrderedDict

from astropy.table import Table
import cPickle as pickle

from pypeline import pype, add_param, depends_on

import syncat.misc as misc
from syn import Syn
from gmm import GaussianMixtureModel

from syncat.misc import struc_array_insert
import syncat.fileio as fileio

import time


@add_param('cat_model', metavar='filename', default='out/syn.pickle', type=str,
                        help='file with catalogue model to load')
@add_param('cond_cat', metavar='filename', default='cond.fits', type=str, help='catalogue with given values')
@add_param('cond_cat_format', metavar='fmt', default='fits', type=str, help='astropy.table format code of catalogue file')
@add_param('cond_cat_columns', metavar='a', default=None, type=str, nargs='*', help='column names for ascii tables without header')
@add_param('cond_params', metavar='name', default=('z',), nargs='*', type=str,
                        help='list of given parameters')
@add_param('hints_file', metavar='filename', default='in/syn_hints.txt', type=str,
                        help='give hints about parameter distributions')
@depends_on(Syn)
class ConditionalMixtureModel(GaussianMixtureModel):
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


    def load_catalog(self, filename, format, columns, hints):
        """ """
        table = fileio.read_catalogue(filename, format=format, columns=columns)
        table_dtype = table.dtype

        table_sub = table

        other_dtypes = {}
        properties = []
        properties_sub = []

        for name in table.columns:
            hit = False
            for skip in self.config['skip']:
                if skip.lower() == name.lower():
                    hit = True
                    self.logger.info("ignoring column '%s' because it includes the string '%s'.", name, skip)
                    break

            if not hit:
                properties.append(name)

                if name not in self.config['cond_params']:
                    properties_sub.append(name)

        table = table[properties]
        table_sub = table_sub[properties_sub]

        if self.logger.isEnabledFor(logging.INFO):
            mesg = ""
            for i, p in enumerate(properties):
                given = ""
                if p in self.config['cond_params']:
                    given = '(conditional)'
                mesg += "\n{:>3} {} {}".format(1 + i, p, given)
            self.logger.info("got these %i columns:%s", len(properties), mesg)

        self.syn = Syn(labels=properties, hints=hints, config=self.config)
        self.syn_sub = Syn(labels=properties_sub, hints=hints, config=self.config)

        dtype = table.dtype
        dtype = misc.append_dtypes(dtype, self.config['add_columns'], table_dtype)

        return table, table_sub, dtype, properties


    def fit(self, filename=None, givenfile=None):
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

            syn_dump, syn_sub_dump = pickle.load(file(self.config['cat_model']))

            self.syn = Syn()
            self.syn.undump(syn_dump)

            self.syn_sub = Syn()
            self.syn_sub.undump(syn_sub_dump)

            self.labels = self.syn.labels
            return

        hints = self.load_hints()

        self.logger.info("loading %s", filename)

        table, table_sub, dtype, properties = self.load_catalog(filename, self.config['input_format'], self.config['input_columns'], hints)

        if self.config['quick']:
            table = table[:50000]
            table_sub = table_sub[:50000]

        self.syn.fit(table, dtype=dtype)
        self.syn_sub.fit(table_sub, dtype=dtype)

        # store column names
        self.labels = properties

        # save catalogue model
        try:
            os.makedirs(os.path.dirname(self.config['cat_model']))
        except OSError:
            pass
        pickle.dump((self.syn.dump(), self.syn_sub.dump()), file(self.config['cat_model'], 'w'))


    def sample(self, cond_cat=None):
        """ Draw samples from the model.

        Parameters
        ----------
        n : float
            number of samples to draw

        Output
        ---------
        ndarray : catalogue data array. shape: (n, dimension)
        """
        if len(self.syn_sub.fit_results) == 0:
            raise SynException("syn.fit() must be called before syn.sample()")

        if cond_cat is None:
            cond_cat = self.config['cond_cat']

        self.logger.info("loading condition parameters from file %s", cond_cat)

        cond_table = fileio.read_catalogue(cond_cat,
            format=self.config['cond_cat_format'],
            columns=self.config['cond_cat_columns'])

        self.logger.info("got these columns: %s", ", ".join(cond_table.columns))

        dtype = misc.append_dtypes(self.syn.dtype, self.config['add_columns'], cond_table.dtype)

        n = len(cond_table)

        self.logger.info("drawing samples: n=%g", n)

        total_count = 0
        for fit in self.syn_sub.fit_results:
            total_count += fit.count

        count_total = 0

        syndata = np.zeros(n, dtype=dtype)
        condparams = cond_table[self.config['cond_params']]
        doneflag = np.zeros(n, dtype=bool)

        count = 0
        loop = 0
        rate = 0
        failrate = 0
        rate_norm = 0
        iter_count = 0
        t0 = time.time()

        while count < n:
            loop += 1
            for fit in self.syn_sub.fit_results:
                iter_count += 1

                # number of samples to draw
                batch = int(np.round(fit.count * n * 1. / total_count))

                sample = fit.sample_me(batch)

                if 'truncate' in self.hints:
                    for name, low, high in self.hints['truncate']:
                        try:
                            i = fit.labels.index(name)
                        except ValueError:
                            continue
                        select = np.where((sample[:, i] > low) & (sample[:, i] < high))

                        n_0 = len(sample)
                        sample = sample[select]
                        #self.logger.debug("truncating %s %s %s: %i -> %i", name, low, high, n_0, len(sample))


                sel, = np.where(doneflag == 0)

                ndraw = min(len(sample), len(sel))

                sel = sel[:ndraw]
                sample = sample[:ndraw]

                p0 = fit.pdf(sample)

                zz = condparams[sel]

                full_fit = self.syn.lookup[fit.hash][0]

                full_sample = np.zeros((ndraw, len(full_fit.labels)))
                for i, name in enumerate(full_fit.labels):
                    if name in zz.columns:
                        full_sample[:, i] = zz[name]
                        continue
                    ind = fit.labels.index(name)
                    full_sample[:, i] = sample[:, ind]

                prob = full_fit.pdf(full_sample)

                ii = p0 > 0
                prob[ii] = prob[ii] / p0[ii]
                pmax = prob.max()

                self.logger.debug("%s pmax %f len %i", fit.hash,  pmax, len(prob))
                if pmax == 0: continue

                r = np.random.uniform(0,1,len(prob))

                too_big = prob > 1
                keep = (prob > r) & (not too_big)

                rate += np.sum(keep)
                rate_norm += len(keep)
                failrate += np.sum(too_big)

                if self.logger.isEnabledFor(logging.DEBUG):
                    if not iter_count%10:
                        message = "{:3.0f} sec - loop {:d} - {:d} samples remaining ({:3.1f}% done) accept rate: {:e} fail rate {:e}".format(time.time()-t0, loop, len(syndata) - count, count*100./len(syndata), rate*1./rate_norm, failrate*1./rate_norm)
                        sys.stderr.write("\r"+message)

                sel = sel[keep]

                full_sample = full_sample[keep]

                tmp = np.zeros(len(full_sample), dtype=syndata.dtype)

                struc_array_insert(tmp, full_sample, full_fit.labels)

                if full_fit.insert_cmd is not None:
                    for column_name, value in full_fit.insert_cmd:
                        tmp[column_name] = value


                syndata[sel] = tmp
                doneflag[sel] = True

                count += len(tmp)

                if count >= n:
                    break

        self.logger.info("loops: %i acceptance rate: %e, fail rate: %e", loop, rate *1./ rate_norm,failrate*1./rate_norm )
        assert count == n

        for instruction, column_name, hint in self.syn.other_dists[::-1]:
            syndata[column_name] = self.syn.sample_dist(instruction, hint, n)

        for name in cond_table.columns:
            if name in self.labels:
                continue
            if name in syndata.dtype.names:
                self.logger.info("Inserting column %s", name)
                syndata[name] = cond_table[name]

        return syndata

