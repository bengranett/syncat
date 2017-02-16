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
from pypelid.survey.mask import Mask
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

		Param('mask_file', default=None, type=str, help='load pypelid mask file to specify survey geometry'),

		Param('fit', default=False, action="store_true",
						help="fit a catalogue model and save to file."),

		Param('syn', default=False, action="store_true",
						help="sample from the model and save output catalogue."),

		Param('density', metavar='x', default=None, type=float, help="number density of objects to synthesize (n/sqr deg)"),

		Param('count', alias='n', metavar='n', default=None, type=float, help="number of objects to synthesize"),

		Param('skip', metavar='name', default=['id', 'skycoord'], nargs='?', help='names of parameters that should be ignored'),

		Param('add_columns', metavar='name', default=[], nargs='?', help='add these columns with zeros if they are present in input catalogue'),

		Param('sample_sky', default=True, action='store_true', help='sample sky coordinates'),

		Param('skycoord_name', default='skycoord', help='column name to use for sky coordinates'),

		Param('verbose', alias='v', default=0, type=int, help='verbosity level'),

		Param('quick', default=False, action='store_true', help='truncate the catalogue for a quick test run'),

		Param('hints_file', default='in/syn_hints.txt', type=str, help='provide hints about parameter distributions'),

		Param('overwrite', default=False, action='store_true', help='overwrite model fit')
	)

	_dependencies = [Syn]

	skycoord_type = np.dtype((np.float64, 2))

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

		if self.config['verbose'] == 0:
			level = logging.CRITICAL
		elif self.config['verbose'] == 1:
			level = logging.INFO
		else:
			level = logging.DEBUG

		logging.basicConfig(level=level)

		self.syn = None

		self.mask = Mask()
		if os.path.exists(self.config['mask_file']):
			self.logger.info("loading mask file %s", self.config['mask_file'])
			self.mask.load(self.config['mask_file'])
		else:
			self.logger.info("no mask file.  full sky")

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


	def load_hints(self):
		""" """
		self.hints = {}

		if os.path.exists(self.config['hints_file']):
			for line in file(self.config['hints_file']):
				line = line.strip()
				if line == "": continue
				if line.startswith("#"): continue

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

	def sample_sky(self, zone=None, nside=None, order=None):
		""" """
		return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=self.config['count'],
															cell=zone, nside=nside))

	def fit_catalogue(self, filename=None):
		""" Initialize the galaxy model. """

		if filename is None:
			filename = self.config['in_cat']

		if os.path.exists(self.config['cat_model']) and not self.config['overwrite']:
			self.logger.info("reading %s", self.config['cat_model'])
			self.syn = Syn(self.config['cat_model'])
			self.labels = self.syn.labels
			return

		hints = self.load_hints()

		self.logger.info("loading %s", filename)

		with CatalogueStore(filename) as store:

			other_dtypes = {}
			properties = []
			for name in store.columns:
				hit = False
				for skip in self.config['skip']:
					if skip.lower() in name.lower():
						hit = True
						self.logger.info("ignoring column '%s' because it includes the string '%s'.", name, skip)
						other_dtypes[name] = np.dtype([(name.encode('ascii'), store.dtype[name])])
						break

				if not hit:
					properties.append(name)

			if self.logger.isEnabledFor(logging.INFO):
				mesg = ""
				for i, p in enumerate(properties):
					mesg += "\n{:>3} {}".format(1 + i, p)
				self.logger.info("got these %i columns:%s", len(properties), mesg)

			self.syn = Syn(labels=properties, hints=hints, config=self.config)

			if self.config['sample_sky']:
				skycoord_dtype = np.dtype([(self.config['skycoord_name'], self.skycoord_type)])

			for zone in store.get_zones():
				batch = store.to_structured_array(columns=properties, zones=[zone])

				dtype = batch.dtype
				for name in self.config['add_columns']:
					try:
						dtype = misc.concatenate_dtypes([dtype, other_dtypes[name]])
					except KeyError:
						pass

				if self.config['sample_sky'] and self.config['skycoord_name'] not in dtype.names:
					dtype = misc.concatenate_dtypes([dtype, skycoord_dtype])

				if self.config['quick']:
					batch = batch[:10000]

				self.syn.fit(batch, dtype=dtype)

				if self.config['quick']:
					break

		# store column names
		self.labels = properties

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
		if self.config['sample_sky']:
			skycoord = self.sample_sky()
			count = len(skycoord)
		else:
			count = self.config['count']

		with CatalogueStore(self.config['out_cat'], 'a', preallocate_file=False) as cat:

			randoms = self.syn.sample(n=count)

			if self.config['sample_sky']:
				randoms[self.config['skycoord_name']] = skycoord

			cat.load(data)

			cat.load_attributes(name='SynCat')

		self.logger.info("output saved to cat %s", self.config['out_cat'])

	def shuffle_catalogue(self, filename=None):
		""" shuffle catalogue. """

		if filename is None:
			filename = self.config['in_cat']

		self.logger.info("loading %s", filename)

		if self.config['overwrite']:
			if os.path.exists(self.config['out_cat']):
				self.logger.info("overwriting existing catalogue: %s", self.config['out_cat'])
				os.unlink(self.config['out_cat'])

		# load full catalogue to shuffle
		data = CatalogueStore(filename).to_structured_array()

		with CatalogueStore(self.config['out_cat'], 'w', preallocate_file=False) as output:

			for zone in np.arange(output._hp_projector.npix):
				# loop through zones in output
				skycoord = self.sample_sky(zone=zone, nside=output._hp_projector.nside)

				if len(skycoord) == 0:
					continue

				data_out = np.random.choice(data, size=len(skycoord), replace=True)

				data_out[self.config['skycoord_name']] = skycoord

				output.load(data_out)

			count = output.count

		self.logger.info("Wrote shuffled catalogue nobj=%i: %s", count, self.config['out_cat'])

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
