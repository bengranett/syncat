""" synrcat

Synthesize a random catalogue.

"""
import sys
import os
import logging
import numpy as np

from pypelid import pypelidobj, add_param, depends_on
from pypelid.utils import misc
from pypelid.vm.syn import Syn
from pypelid.sky.catalogue_store import CatalogueStore
from pypelid.survey.mask import Mask
from pypelid.utils.config import Config
from pypelid.utils import sphere

import time

SHUFFLE_MODE = 'shuffle'
GMM_MODE = 'gmm'
ZDIST_MODE = 'radial'
skycoord_type = np.dtype((np.float64, 2))

@add_param('cat_model', metavar='filename', default='out/syn.pickle', type=str,
						help='file with catalogue model to load')
@add_param('fit', default=False, action="store_true",
						help="fit a catalogue model and save to file.")
@add_param('hints_file', metavar='filename', default='in/syn_hints.txt', type=str,
						help='give hints about parameter distributions')
@depends_on(Syn)
class GaussianMixtureModel(pypelidobj):
	""" """

	def __init__(self, config={}, mask=None, **kwargs):
		""" """
		self._parse_config(config, **kwargs)
		self._setup_logging()

		self.load_hints()

		self.mask = mask

		self.syn = None

	def sample_sky(self, zone=None, nside=None, order=None):
		""" """
		return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=self.config['count'],
															cell=zone, nside=nside))

	def load_hints(self):
		""" """
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
				skycoord_dtype = np.dtype([(self.config['skycoord_name'], skycoord_type)])

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

	def sample(self):
		""" """
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
			randoms[self.config['skycoord_name']] = skycoord

		with CatalogueStore(self.config['out_cat'], 'a', preallocate_file=False) as cat:

			cat.load(randoms)

			cat.load_attributes(name='SynCat', method=self.config['method'])

			self.logger.debug("count: %i", cat.count)

		self.logger.info("output saved to cat %s", self.config['out_cat'])

		return randoms

@add_param('zdist_file', metavar='filename', default='in/zdist.txt', type=str, help='path to file specifying redshift distribution')
class Radial(pypelidobj):
	""" """

	def __init__(self, config={}, mask=None, **kwargs):
		""" """
		self._parse_config(config, **kwargs)
		self._setup_logging()

		self.zdist = None
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

		self.zdist = (zz, nz)
		self.logger.info("Loaded redshift distribution file %s.", filename)

	def sample_sky(self, zone=None, nside=None, order=None):
		""" """
		return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=self.config['count'],
															cell=zone, nside=nside))

	def fit(self):
		""" """
		pass

	def sample(self):
		""" Draw redshift from the distribution. """
		if self.zdist is None:
			self.load_zdist()

		zz, nz = self.zdist


class Shuffle(pypelidobj):
	""" """

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


@add_param("in_cat", metavar='filename', default='in/galaxies.pypelid.hdf5', type=str, help='input catalog')
@add_param("out_cat", metavar='filename', default='out/syn.pypelid.hdf5', type=str, help='catalog file to write')
@add_param('mask_file', metavar='filename', default=None, type=str, help='load pypelid mask file to specify survey geometry')
@add_param('method', default=GMM_MODE, type=str, choices=(GMM_MODE, SHUFFLE_MODE, ZDIST_MODE),
				help='method to generate catalogue (gmm, shuffle, radial)')
@add_param('sample', default=False, action="store_true",
				help="generate samples and save output catalogue.")
@add_param('density', metavar='x', default=None, type=float, help="number density of objects to synthesize (n/sqr deg)")
@add_param('count', alias='n', metavar='n', default=None, type=float, help="number of objects to synthesize")
@add_param('skip', metavar='name', default=['id', 'skycoord'], nargs='?', help='names of parameters that should be ignored')
@add_param('add_columns', metavar='name', default=[], nargs='?', help='add these columns with zeros if they are present in input catalogue')
@add_param('sample_sky', default=True, action='store_true', help='sample sky coordinates')
@add_param('skycoord_name', metavar='name', default='skycoord', help='column name of sky coordinates')
@add_param('verbose', alias='v', default=0, type=int, help='verbosity level')
@add_param('quick', default=False, action='store_true', help='truncate the catalogue for a quick test run')
@add_param('overwrite', default=False, action='store_true', help='overwrite model fit')
@depends_on(GaussianMixtureModel, Shuffle, Radial)
class SynCat(pypelidobj):
	""" SynCat """

	modes = {
		GMM_MODE: GaussianMixtureModel,
		SHUFFLE_MODE: Shuffle,
		ZDIST_MODE: Radial,
	}

	def __init__(self, config={}, **kwargs):
		""" """
		self._parse_config(config, **kwargs)
		self._setup_logging()
		self.logger.info("Starting SynCat")

		self.load_mask()

		self.synthesizer = self.modes[config['method']](self.config, self.mask)

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

	def run(self):
		""" Run SynCat pipeline.

		Parameters
		----------

		Returns
		-------
		"""
		if self.config['fit']:
			self.logger.info("Starting fit")
			self.synthesizer.fit()

		if self.config['sample']:
			self.logger.info("Starting sampling")
			self.synthesizer.sample()


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
