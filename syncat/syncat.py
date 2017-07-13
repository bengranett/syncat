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

import errors

import time

SHUFFLE_MODE = 'shuffle'
GMM_MODE = 'gmm'
ZDIST_MODE = 'radial'


@add_param("in_cat", metavar='filename', default='in/galaxies.pypelid.hdf5', type=str, help='input catalog')
@add_param("input_format", metavar='fmt', default='fits', type=str, help='input catalog format')
@add_param('input_columns', metavar='a', default='', type=str, nargs='*', help='column names for ascii tables without header')
@add_param("out_cat", metavar='filename', default='randoms.fits', type=str, help='catalog file to write')
@add_param("output_format", metavar='fmt', default='fits', type=str, help='output catalog format')
@add_param('mask_file', metavar='filename', default='', type=str, help='load pypelid mask file to specify survey geometry')
@add_param('method', default=GMM_MODE, type=str, choices=(GMM_MODE, SHUFFLE_MODE, ZDIST_MODE),
				help='method to generate catalogue (gmm, shuffle, radial)')
@add_param('sample', default=False, type='bool',
				help="generate samples and save output catalogue.")
@add_param('fit', default=False, type='bool',
						help="fit a catalogue model and save to file.")
@add_param('density', metavar='x', default=-1, type=float, help="number density of objects to synthesize (n/sqr deg)")
@add_param('count', alias='n', metavar='n', default=-1, type=float, help="number of objects to synthesize")
@add_param('skip', metavar='name', default=['id', 'num', 'skycoord', 'alpha', 'delta'], 
				nargs='*', help='names of parameters that should be ignored')
@add_param('add_columns', metavar='name', default=[], nargs='*', help='add these columns with zeros if they are present in input catalogue')
@add_param('sample_sky', default=True, type='bool', help='sample sky coordinates')
@add_param('skycoord_name', metavar='name', default=('alpha', 'delta'), nargs='*', help='column name(s) of sky coordinates')
@add_param('verbose', alias='v', default=0, type=int, help='verbosity level')
@add_param('quick', default=0, type=int, help='truncate the catalogue for a quick test run')
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

		self.check_params()

		if mask is None:
			mask = self.load_mask()

		self.synthesizer = self.modes[config['method']](self.config, mask)


	def check_params(self):
		""" """
		if self.config['count'] <= 0:
			self.config['count'] = None

		if self.config['density'] <= 0:
			self.config['density'] = None

	@staticmethod
	def check_config(config):
		""" """
		if config['count'] <= 0 and config['density'] <= 0:
			raise ValueError('Nothing to sample! Must give density or count.')

		if config['count'] is None and config['density'] is None:
			raise ValueError('Nothing to sample! Must give density or count.')


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

		if self.config['sample'] or sample:
			if os.path.exists(self.config['out_cat']) and not self.config['overwrite']:
				self.logger.critical("Stopping early because output file exists and should not be over-written: %s", self.config['out_cat'])
				return

		if self.config['fit']:
			self.logger.info("Starting fit")
			self.synthesizer.fit()
			done = True

		if self.config['sample'] or sample:
			self.logger.info("Starting sampling")
			try:
				data = self.synthesizer.sample()
			except errors.NoPoints:
				data = []
				self.logger.warning("No points were sampled!  Perhaps mask does not align with pointings.")
			self.write_cat(data)
			done = True

		if not done:
			self.logger.info("run() had nothing to do and so did nothing")

	def write_cat(self, data):
		""" """
		table = Table(data=data)
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

	logging.getLogger().setLevel(level)

	logger = logging.getLogger(__name__)

	banner = "~"*70+"\n{:^70}\n".format(tagline)+"~"*70
	logger.info("\n" + banner)
	logger.info(config)
	# Run code
	try:
		S = SynCat(config=config)
		S.run()
	except Exception as e:
		raise
		print >>sys.stderr, traceback.format_exc()

if __name__ == '__main__':
	main()
