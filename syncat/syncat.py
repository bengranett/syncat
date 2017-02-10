""" Visibility mask

Pipeline to build the visibility mask in the form of a weighted random
catalogue.

"""
import sys
import os
import logging
import numpy as np

from pypelid.utils import misc
from pypelid.vm.syn import Syn
from pypelid.sky.catalogue_store import CatalogueStore
from pypelid.utils.config import Config, Defaults, Param
from pypelid.utils import sphere

import time


class SynCat(object):
    """ SynCat """
    logger = logging.getLogger(__name__)

    # default parameters will be copied in if missing
    _default_params = Defaults(

        Param("in_cat", metavar='filename', default='in/galaxies.pypelid.hdf5', type=str, help='input catalog'),

        Param("out_cat", metavar='filename', default='out/syn.pypelid.hdf5', type=str, help='catalog file to write'),

        Param("cat_model", metavar='filename', default='out/syn.pickle', type=str, help='file with catalogue model to load'),

        Param('fit', default=False, action="store_true",
                        help="fit a catalogue model and save to file."),

        Param('syn', default=False, action="store_true",
                        help="sample from the model and save output catalogue."),

        Param('nsyn', alias='n', metavar='n', default=1e6, type=float, help="number of objects to synthesize"),

        Param('skip', metavar='name', default=['id', 'skycoord'], nargs='?', help='names of parameters that should be ignored'),

        Param('verbose', alias='v', default=0, type=int, help='verbosity level'),

        Param('quick', default=False, action='store_true', help='truncate the catalogue for a quick test run')
    )

    _dependencies = [Syn]

    def __init__(self, config={}, **args):
        """ """
        self.config = config

        # merge key,value arguments and config dict
        for key, value in args.items():
            self.config[key] = value

        # Add any parameters that are missing
        for key, def_value in self._default_params.items():
            try:
                self.config[key]
            except KeyError:
                self.config[key] = def_value

        self.syn = None

    @staticmethod
    def check_config(config):
        """ """
        pass

    def fit_catalogue(self):
        """ Initialize the galaxy model. """

        if os.path.exists(self.config['cat_model']):
            self.logger.info("reading %s", self.config['cat_model'])
            self.syn = Syn(self.config['cat_model'])
            return

        self.logger.info("loading %s", self.config['in_cat'])

        with CatalogueStore(self.config['in_cat']) as cat:

            properties = []
            for name in cat._h5file.get_columns():
                hit = False
                for skip in self.config['skip']:
                    if skip.lower() in name.lower():
                        hit = True
                        self.logger.info("ignoring column '%s' because it includes the string '%s'.", name, skip)
                        break
                if not hit:
                    properties.append(name)

            if self.logger.isEnabledFor(logging.INFO):
                mesg = ""
                for i, p in enumerate(properties):
                    mesg += "\n{:>3} {}".format(1 + i, p)
                self.logger.info("got these %i columns:%s", len(properties), mesg)

            self.syn = Syn(labels=properties, config=self.config)

            for zone in cat:
                batch = []
                for prop in properties:
                    batch.append(np.array(getattr(zone, prop)))

                batch = np.transpose(batch)

                if self.config['quick']:
                    batch = batch[:10000]

                self.syn.fit(batch)

                if self.config['quick']:
                    break

        # save catalogue model
        self.syn.save(self.config['cat_model'])

    def build_catalogue(self):
        """ Build random catalogue.

        Parameters
        ----------

        Returns
        -------
        dictionary : catalog
        """
        with CatalogueStore(self.config['out_cat'], 'a', preallocate_file=False) as cat:

            randoms = self.syn.sample(n=self.config['nsyn'])

            data = {}
            for i, name in enumerate(self.syn.labels):
                data[name] = randoms[:, i]

            data['skycoord'] = np.transpose(sphere.sample_sphere(len(randoms)))

            cat.load(data)

            cat.load_attributes(name='SynCat')

        self.logger.info("output saved to cat %s", self.config['out_cat'])

    def run(self):
        """ Run visibilty mask pipeline.

        Parameters
        ----------

        Returns
        -------
        """
        if self.config['fit']:
            self.fit_catalogue()

        if self.config['syn']:
            if self.syn is None:
                if not os.path.exists(self.config['cat_model']):
                    self.logger.error("Cannot load catalogue model.  Files does not exist: %s", self.config['cat_model'])
                    sys.exit(1)
                self.syn = Syn(self.config['cat_model'])
            self.build_catalogue()


def main(args=None):
    """ Main routine so can have other entry points """

    # access configuration:
    tagline = 'SynCat synthesizes catalogs'

    config = Config([SynCat], description=tagline)

    if config['verbose'] == 0:
        level = logging.CRITICAL
    elif config['verbose'] == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(level=level)

    banner = "~"*70+"\n{:^70}\n".format(tagline)+"~"*70
    logging.info("\n" + banner)
    logging.info(config)

    # Get code version
    git = misc.GitEnv()

    # Run code
    try:
        S = SynCat(config)
        S.run()
    except Exception as e:
        raise
        print >>sys.stderr, traceback.format_exc()
        logging.debug(misc.GitEnv())

if __name__ == '__main__':
    main()
