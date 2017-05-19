""" synrcat

Synthesize a random catalogue.

"""
import sys
import os
import logging
import numpy as np

from astropy.table import Table

from pypeline import pype, Config, add_param, depends_on

from minimask.mask import Mask
from minimask.mask import sphere

import methods.gmm as gmm
import methods.radial as radial
import methods.shuffle as shuffle

import time

SHUFFLE_MODE = 'shuffle'
GMM_MODE = 'gmm'
ZDIST_MODE = 'radial'


@add_param("in_cat", metavar='filename', default='in/galaxies.pypelid.hdf5', type=str, help='input catalog')
@add_param("input_format", metavar='fmt', default=None, type=str, help='input catalog format')
@add_param("out_cat", metavar='filename', default='randoms.txt', type=str, help='catalog file to write')
@add_param("output_format", metavar='fmt', default='ascii', type=str, help='output catalog format')
@add_param('mask_file', metavar='filename', default=None, type=str, help='load pypelid mask file to specify survey geometry')
@add_param('method', default=GMM_MODE, type=str, choices=(GMM_MODE, SHUFFLE_MODE, ZDIST_MODE),
				help='method to generate catalogue (gmm, shuffle, radial)')
@add_param('sample', default=False, type='bool',
				help="generate samples and save output catalogue.")
@add_param('fit', default=False, type='bool',
						help="fit a catalogue model and save to file.")
@add_param('density', metavar='x', default=None, type=float, help="number density of objects to synthesize (n/sqr deg)")
@add_param('count', alias='n', metavar='n', default=None, type=float, help="number of objects to synthesize")
@add_param('skip', metavar='name', default=['id', 'num', 'skycoord', 'alpha', 'delta'], 
				nargs='*', help='names of parameters that should be ignored')
@add_param('add_columns', metavar='name', default=[], nargs='*', help='add these columns with zeros if they are present in input catalogue')
@add_param('sample_sky', default=True, type='bool', help='sample sky coordinates')
@add_param('skycoord_name', metavar='name', default=('alpha', 'delta'), nargs='*', help='column name(s) of sky coordinates')
@add_param('verbose', alias='v', default=0, type=int, help='verbosity level')
@add_param('quick', default=False, type='bool', help='truncate the catalogue for a quick test run')
@add_param('overwrite', default=False, type='bool', help='overwrite model fit')
@depends_on(gmm.GaussianMixtureModel, shuffle.Shuffle, radial.Radial)
class SynCat(pype):
	""" SynCat """

	modes = {
		GMM_MODE: gmm.GaussianMixtureModel,
		SHUFFLE_MODE: shuffle.Shuffle,
		ZDIST_MODE: radial.Radial,
	}

	def __init__(self, mask=None, config={}, **kwargs):
		""" """
		self._parse_config(config, **kwargs)
		self._setup_logging()
		self.logger.info("Starting SynCat")

		if mask is None:
			mask = self.load_mask()

		self.synthesizer = self.modes[config['method']](self.config, mask)

	@staticmethod
	def check_config(config):
		""" """
		if config['density'] is None and config['count'] is None:
			raise Exception("must specify either density or count")

		if config['density'] is not None and config['count'] is not None:
			raise Exception("must specify either density or count (not both)")

		if config['density'] is not None and config['mask_file'] is None:
			raise Exception("a mask file is needed for density mode")

		if config['density'] is not None and config['sample_sky'] is None:
			raise Exception("sample_sky must be True in density mode")

	def load_mask(self):
		""" """
		self.mask = Mask()

		if self.config['mask_file'] and os.path.exists(self.config['mask_file']):
			self.logger.info("loading mask file %s", self.config['mask_file'])
			self.mask.load(self.config['mask_file'])
		else:
			self.logger.info("no mask file.  full sky")

		return self.mask

	def run(self, sample=True):
		""" Run SynCat pipeline.

		Parameters
		----------

		Returns
		-------
		"""
		done = False

		if self.config['fit']:
			self.logger.info("Starting fit")
			self.synthesizer.fit()
			done = True

		if self.config['sample'] or sample:
			self.logger.info("Starting sampling")
			data = self.synthesizer.sample()
			self.write_cat(data)

			done = True

		if not done:
			self.logger.info("run() has nothing to do and so did nothing")

	def write_cat(self, data):
		""" """
		table = Table(data=data)
		print self.config
		table.write(self.config['out_cat'], format=self.config['output_format'],
			overwrite=self.config['overwrite'])
		self.logger.info("wrote catalogue %s", self.config['out_cat'])

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

	# Run code
	try:
		S = SynCat(config)
		S.run(False)
	except Exception as e:
		raise
		print >>sys.stderr, traceback.format_exc()

if __name__ == '__main__':
	main()
