import sys, os
import numpy as np
import logging
import time

import cPickle as pickle
from sklearn import mixture, decomposition

from pypelid.utils.config import Defaults, Param
from pypelid.utils.misc import increment_path


class FitResults(object):
	""" Data structure to store parameters of the fit. """
	def __init__(self, **kwargs):
		""" """
		self.__dict__.update(kwargs)


class Syn:
	""" """
	logger = logging.getLogger(__name__)

	_default_params = Defaults(
		Param('ncomponents', metavar='n', default=10, type=int, help="Number of Gaussian components to mix"),
		Param('batch_size', metavar='n', default=10000, type=int, help="Fit in batches of this size", hidden=True),
		Param('nloops', metavar='n', default=1, type=int, help="Number of times to repeat fit and take the best one"),
		Param('apply_pca', default=False, action='store_true', help="Preprocess with PCA", hidden=True),
		Param('covariance_mode', metavar='mode', default='diag', choices=['diag', 'full', 'tied', 'spherical'],
			help="Mode for modelling the Gaussian components.", hidden=True),
		Param('min_fit_size',metavar='n',  default=50, type=int, help="Minimimum size of dataset to do Gaussian fit", hidden=True),
		Param('discreteness_count', metavar='n', default=20, type=int, help="If count of unique values is less than this, array is discrete", hidden=True),
		Param('special_values', metavar='x', default=[0, -99, float('nan')], nargs='?', help='values that will be treated as discrete.'),
		Param('log_crit', metavar='x', default=1, help='will try a log transform if the dynamic range is greater than this and all positive.', hidden=True),
	)

	def __init__(self, loadfile=None, labels=None, config={}, **kwargs):
		""" Build synthetic datasets. """
		self.config = config

		# merge key,value arguments and config dict
		for key, value in kwargs.items():
			self.config[key] = value

		# Add any parameters that are missing
		for key, def_value in self._default_params.items():
			try:
				self.config[key]
			except KeyError:
				self.config[key] = def_value

		self.labels = labels
		self.loadfile = loadfile

		self.fit_results = []

		self.load(self.loadfile)

	def load(self, path):
		""" Load a catalogue model file from a previous run. """
		if path is not None and os.path.exists(path):
			self.labels, self.fit_results = pickle.load(file(path))
			self.logger.info("Lodaed %s, got %i fit results.", path, len(self.fit_results))

	def save(self, path=None):
		""" Save the catalogue model file. """
		if path is None:
			path = self.loadfile

		dir = os.path.dirname(path)
		if dir != '':
			if not os.path.exists(dir):
				os.mkdir(dir)

		path = increment_path(path)

		if os.path.exists(path):
			self.logger.warning("File exists: %s.  Will not overwrite.", path)
		else:
			pickle.dump((self.labels, self.fit_results), file(path, 'w'))
			self.logger.info("wrote %s", path)

	def _fit(self, data, k=30, loops=10):
		""" Fit the data with a Gaussian mixture model.
		Uses the sklearn.mixture.GMM routine.

		Parameters
		----------
		data : ndarray
		k : int
			number of components
		loops : int
			try a number of times and take the solution that maximizes the Akaike
			criterion.

		Output
		------
		None

		"""
		# quick check on the array orientation
		count, dim = data.shape
		assert dim < count

		# whiten the data
		mu = np.mean(data, axis=0)
		sig = np.std(data, axis=0)

		# chcek that means are computed in each dim
		assert len(mu) == dim

		data_w = (data - mu) / sig

		if self.config['apply_pca']:
			# PCA rotation
			pca = decomposition.PCA()
			data_w = pca.fit_transform(data_w)
		else:
			pca = None

		best = None
		best_aic = None
		aic = None
		for i in range(loops):
			G = mixture.GMM(n_components=k, covariance_type=self.config['covariance_mode'])
			G.fit(data_w)
			if not G.converged_:
				continue

			if loops > 1:
				aic = G.aic(data_w)

			if best_aic is None or aic < best_aic:
				best = G
				best_aic = aic

		if best is None:
			raise SynException("No good GMM fit was found!")

		G = best

		fit_results = FitResults(mu=mu, sigma=sig, gmm=G, best_aic=best_aic,
								pca=pca, count=count)

		return fit_results

	def single_fit(self, data, insert_cmd=None):
		""" Fit the data with a Gaussian mixture model.
		Uses the sklearn.mixture.GMM routine.

		Parameters
		----------
		data : ndarray
		k : int
			number of components
		loops : int
			try a number of times and take the solution that maximizes the Akaike
			criterion.

		Output
		------
		None

		"""
		n, dim = data.shape
		assert dim < n

		# compute number of batches
		nbatch = max(1, int(n * 1. / self.config['batch_size']))
		assert nbatch >= 1

		# log columns that are positive
		logtransform = []
		for i in range(data.shape[1]):
			low, high = data[:, i].min(), data[:, i].max()
			assert high > low
			if low > 0 and high / low > self.config['log_crit']:
				data[:, i] = np.log(data[:, i])

				logtransform.append((np.exp, i))

		bins = np.linspace(0, n, nbatch + 1).astype(int)

		dt = 0
		count = 0
		for i, j in zip(bins[:-1], bins[1:]):
			if count > 0:
				t = dt / count
			else:
				t = 0

			sub = data[i:j]

			t0 = time.time()
			fit = self._fit(sub, self.config['ncomponents'], self.config['nloops'])
			dt += time.time() - t0
			count += 1

			fit.insert_cmd = insert_cmd
			fit.transform = logtransform

			self.fit_results.append(fit)

	def _branch_fit(self, data, column=0, insert=[]):
		""" """
		n, dim = data.shape
		assert dim < n

		if column == dim:
			self.fits_to_run.append((data, insert))
			return

		values = np.unique(data[:, column])
		if len(values) < self.config['discreteness_count']:
			discrete = True
			special_values = values
		else:
			discrete = False
			special_values = self.config['special_values']

		unmatched = np.ones(n, dtype=bool)

		for value in special_values:
			matches = np.isclose(data[:, column], value)
			unmatched[matches] = False

			if np.sum(matches) > self.config['min_fit_size']:
				data_z = np.delete(data[matches], column, axis=1)
				ins_cmd = (column, value)
				self._branch_fit(data_z, column, insert=insert + [ins_cmd])

		# what remains after the discrete values
		if np.sum(unmatched) > self.config['min_fit_size']:
			data_nz = data[unmatched]
			self._branch_fit(data[unmatched], column + 1, insert=insert)

	def progress_bar(self):
		""" """
		if self.config['verbose'] < 1:
			return

		try:
			t = time.time() - self._t0
		except AttributeError:
			t = 0
			self._count = 0
			self._t0 = time.time()

		self._count += 1
		tot = t / self._count * self._fit_count
		remaining = int(tot - t)
		if t < 30:
			remaining = "?"
		sys.stderr.write("\r Working hard! %i of %i fits, elapsed time: %i sec, remaining: approx. %s sec." % (self._count, self._fit_count, t, remaining))

	def fit(self, data):
		""" Fit the data with a Gaussian mixture model.
		The data is partitioned to handle discrete columns and special values.

		Parameters
		----------
		data : ndarray
			catalogue data to fit. shape should be (nobj, dimension)

		Output
		------
		None
		"""
		self.fits_to_run = []
		self._branch_fit(data)
		self._fit_count = len(self.fits_to_run)

		while True:
			self.progress_bar()
			try:
				d, ins_cmd = self.fits_to_run.pop()
			except IndexError:
				break
			self.single_fit(d, insert_cmd=ins_cmd)

	def sample(self, n=1e5):
		""" Draw samples from the model.

		Parameters
		----------
		n : float
			number of samples to draw

		Output
		---------
		ndarray : catalogue data array. shape: (n, dimension)
		"""

		if len(self.fit_results) == 0:
			raise SynException("syn.fit() must be called before syn.sample()")

		self.logger.info("drawing samples: n=%g", n)

		total_count = 0
		for fit in self.fit_results:
			total_count += fit.count

		data_out = []
		for fit in self.fit_results:

			# number of samples to draw
			batch = int(fit.count * n * 1. / total_count)

			syndata = fit.gmm.sample(batch)

			if fit.pca is not None:
				syndata = fit.pca.inverse_transform(syndata)

			syndata = syndata * fit.sigma + fit.mu

			for func, i in fit.transform:
				syndata[:, i] = func(syndata[:, i])

			if fit.insert_cmd is not None:
				for cmd in fit.insert_cmd[::-1]:
					syndata = np.insert(syndata, *cmd, axis=1)

			data_out.append(syndata)

		return np.vstack(data_out)

class SynException(Exception):
	pass
