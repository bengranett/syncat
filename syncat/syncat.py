""" synrcat

Synthesize a random catalogue.

"""
import sys
import os
import logging
import numpy as np
from collections import OrderedDict

from astropy.table import Table

from pypeline import pype, Config, add_param, depends_on

import pypelid.utils.misc as misc
from syn import Syn
from sample_dist import sample_dist
from minimask.mask import Mask
from minimask.mask import sphere

import time

SHUFFLE_MODE = 'shuffle'
GMM_MODE = 'gmm'
ZDIST_MODE = 'radial'


@add_param('cat_model', metavar='filename', default='out/syn.pickle', type=str,
						help='file with catalogue model to load')
@add_param('hints_file', metavar='filename', default='in/syn_hints.txt', type=str,
						help='give hints about parameter distributions')
@depends_on(Syn)
class GaussianMixtureModel(pype):
	""" SynCat mode to generate random catalogue by sampling from a gaussian mixture model."""

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
			dim = len(misc.ensurelist(self.config['skycoord_name']))

			if dim == 1:
				skycoord_dtype = np.dtype([(self.config['skycoord_name'], np.dtype((np.float64, 2)))])
			elif dim == 2:
				alpha, delta = self.config['skycoord_name']
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
			dim = len(misc.ensurelist(self.config['skycoord_name']))

			if dim == 1:
				randoms[self.config['skycoord_name']] = skycoord
			else:
				for i in range(dim):
					randoms[self.config['skycoord_name'][i]] = skycoord[:,i]

		return randoms


@add_param('zdist_file', metavar='filename', default='in/zdist.txt', type=str,
	help='path to file specifying redshift distribution')
@add_param('zdist_interp', metavar='name', default='pchip', choices=('linear','pchip'), type=str,
	help='method to interpolate cumulative distribution function')
class Radial(pype):
	""" SynCat mode to generate a random catalogue by drawing redshift from a distribution."""

	def __init__(self, config={}, mask=None, zdist=None, **kwargs):
		""" """
		self._parse_config(config, **kwargs)
		self._setup_logging()

		self.zdist = zdist
		if zdist is None:
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

		step = zz[1] - zz[0]
		bin_edges = np.arange(len(zz) + 1) * step + zz[0] - step / 2.

		self.zdist = (bin_edges, nz)
		self.logger.info("Loaded redshift distribution file %s.", filename)

	def sample_sky(self, zone=None, nside=None, order=None):
		""" """
		return np.transpose(self.mask.draw_random_position(dens=self.config['density'], n=int(self.config['count']),
															cell=zone, nside=nside))

	def fit(self):
		""" """
		pass

	def sample(self):
		""" Draw samples from the radial selection function."""
		if self.config['overwrite']:
			if os.path.exists(self.config['out_cat']):
				self.logger.info("overwriting existing catalogue: %s", self.config['out_cat'])
				os.unlink(self.config['out_cat'])

		bin_edges, bin_counts = self.zdist

		sampler = sample_dist(bin_edges, bin_counts)

		skycoord = self.sample_sky()

		redshift = sampler(len(skycoord))

		data_out = OrderedDict({'z': redshift})

		dim = len(misc.ensurelist(self.config['skycoord_name']))
		if dim == 1:
			data_out[self.config['skycoord_name']] = skycoord
		else:
			for i in range(dim):
				data_out[self.config['skycoord_name'][i]] = skycoord[:, i]

		print data_out.keys()
		data_out = misc.dict_to_structured_array(data_out)

		self.logger.info("Wrote radial random catalogue nobj=%i: %s", len(data_out), self.config['out_cat'])

		return data_out


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

		dim = len(misc.ensurelist(self.config['skycoord_name']))
		if dim == 1:
			data_out[self.config['skycoord_name']] = skycoord
		else:
			for i in range(dim):
				data_out[self.config['skycoord_name'][i]] = skycoord[:, i]

		self.logger.info("Wrote shuffled catalogue nobj=%i: %s", len(data_out), self.config['out_cat'])

		return data_out


@add_param("in_cat", metavar='filename', default='in/galaxies.pypelid.hdf5', type=str, help='input catalog')
@add_param("input_format", metavar='fmt', default=None, type=str, help='input catalog format')
@add_param("out_cat", metavar='filename', default='out/syn.pypelid.hdf5', type=str, help='catalog file to write')
@add_param("output_format", metavar='fmt', default=None, type=str, help='output catalog format')
@add_param('mask_file', metavar='filename', default=None, type=str, help='load pypelid mask file to specify survey geometry')
@add_param('method', default=GMM_MODE, type=str, choices=(GMM_MODE, SHUFFLE_MODE, ZDIST_MODE),
				help='method to generate catalogue (gmm, shuffle, radial)')
@add_param('sample', default=False, action="store_true",
				help="generate samples and save output catalogue.")
@add_param('fit', default=False, action="store_true",
						help="fit a catalogue model and save to file.")
@add_param('density', metavar='x', default=None, type=float, help="number density of objects to synthesize (n/sqr deg)")
@add_param('count', alias='n', metavar='n', default=None, type=float, help="number of objects to synthesize")
@add_param('skip', metavar='name', default=['id', 'num', 'skycoord', 'alpha', 'delta'], 
				nargs='?', help='names of parameters that should be ignored')
@add_param('add_columns', metavar='name', default=[], nargs='?', help='add these columns with zeros if they are present in input catalogue')
@add_param('sample_sky', default=True, action='store_true', help='sample sky coordinates')
@add_param('skycoord_name', metavar='name', default=('alpha', 'delta'), nargs='?', help='column name(s) of sky coordinates')
@add_param('verbose', alias='v', default=0, type=int, help='verbosity level')
@add_param('quick', default=False, action='store_true', help='truncate the catalogue for a quick test run')
@add_param('overwrite', default=False, action='store_true', help='overwrite model fit')
@depends_on(GaussianMixtureModel, Shuffle, Radial)
class SynCat(pype):
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
		table.write(self.config['out_cat'], format=self.config['output_format'],
			overwrite=self.config['overwrite'])

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
		S.run(False)
	except Exception as e:
		raise
		print >>sys.stderr, traceback.format_exc()
		logging.debug(misc.GitEnv())

if __name__ == '__main__':
	main()
