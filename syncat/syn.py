import sys, os
import numpy as np
import logging
import time

import cPickle as pickle
from sklearn import mixture, decomposition

from pypelid.utils.config import Defaults, Param
from pypelid.utils.misc import increment_path, flatten_struc_array
import pypelid.utils.misc as misc


def struc_array_insert(arr, data, labels, index=0, truncate=True):
	""" Insert data into a structured array.

	Parameters
	----------
	arr : numpy structured array
		structured array to insert into
	data : numpy ndarray
		array of values to insert
	labels : sequence
		column names corresponding to data

	Returns
	---------
	int : number of elements inserted
	"""
	assert len(data.shape) == 2

	j = index + len(data)
	if j > len(arr):
		if not truncate:
			raise ValueError("data array too large to fit in structured array.")
		j = len(arr)
		sub = data[: j - index]
	else:
		sub = data

	for i, name in enumerate(labels):
		arr[name][index:j] = sub[:, i]

	return len(sub)


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

	def __init__(self, loadfile=None, labels=None, hints={}, config={}, **kwargs):
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

		self.hints = hints

		self.load(self.loadfile)

	def load(self, path):
		""" Load a catalogue model file from a previous run. """
		if path is not None and os.path.exists(path):
			self.labels, self.fit_results, self.other_dists, self.hints = pickle.load(file(path))
			self.logger.info("Lodaed %s, got %i fit results.", path, len(self.fit_results))

	def save(self, path=None):
		""" Save the catalogue model file. """
		if path is None:
			path = self.loadfile

		dir = os.path.dirname(path)
		if dir != '':
			if not os.path.exists(dir):
				os.mkdir(dir)

		# path = increment_path(path)

		# if os.path.exists(path):
			# self.logger.warning("File exists: %s.  Will not overwrite.", path)
		# else:
		pickle.dump((self.labels, self.fit_results, self.other_dists, self.hints, self.dtype), file(path, 'w'))
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

	def single_fit(self, data, labels=None, insert_cmd=None):
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
				# self.logger.info("\nlog transform %s %s %s", labels[i], low, high)
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

			fit.labels = labels
			fit.insert_cmd = insert_cmd
			fit.transform = logtransform

			self.fit_results.append(fit)

	def _branch_fit(self, data, labels, column=0, insert=[]):
		""" """
		n, dim = data.shape
		assert dim < n

		if column == dim:
			self.fits_to_run.append((data, labels, insert))
			return

		# determine if the array has discrete values
		values = np.unique(data[:, column])
		if len(values) < self.config['discreteness_count']:
			discrete = True
			special_values = values
		else:
			# otherwise us a list of default special values eg 0, -99, nan,...
			discrete = False
			special_values = self.config['special_values']

		unmatched = np.ones(n, dtype=bool)

		# compute absolute tolerance for equality
		mu = np.abs(data[:, column]).mean()
		tol = mu / 1e5
		for value in special_values:
			# find special values in the array
			matches = np.isclose(data[:, column], value, equal_nan=True, atol=tol)
			unmatched[matches] = False

			if np.sum(matches) > self.config['min_fit_size']:
				# there is a sufficient number left to fit
				# remove column with discrete value
				data_z = np.delete(data[matches], column, axis=1)
				labels_cut = labels[:]
				name = labels_cut.pop(column)
				ins_cmd = (name, value)
				self._branch_fit(data_z, labels_cut, column, insert=insert + [ins_cmd])

		# what remains after the discrete values
		if np.sum(unmatched) > self.config['min_fit_size']:
			data_nz = data[unmatched]
			self._branch_fit(data[unmatched], labels[:], column + 1, insert=insert)

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
		elapsed = "%i sec"%t
		if t < 30:
			remaining = "?"
			elapsed = "%i sec"%t
			mesg = 'starting up...'
		elif t < 60:
			elapsed = "%g min"%(np.round(t/60.,1))
			mesg = 'working hard!'
		elif t < 180:
			elapsed = "%g min"%(np.round(t/60.,1))
			mesg = 'still working...'
		else:
			elapsed = "%g min"%(np.round(t/60.,1))
			mesg = '%i bottles of beer on the wall...'%(self._fit_count - self._count)

		sys.stderr.write("\r%s done %i of %i fits, elapsed time: %s, remaining: approx. %s sec." % (mesg, self._count, self._fit_count, elapsed, remaining))

	def process_hints(self, data):
		""" """
		_labels = self.labels[:]

		self.other_dists = []

		for instruction, hints in self.hints.items():
			if instruction == 'uniform':
				for hint in hints:
					key = hint[0]
					i = _labels.index(key)
					_labels.pop(i)
					data = np.delete(data, i, axis=1)
					self.other_dists.append((instruction, key, hint))
			elif instruction == 'smooth':
				for hint in hints:
					key = hint[0]
					sigma = float(hint[1])
					i = _labels.index(key)
					self.logger.info("Smoothing %s sigma=%f", key, sigma)
					data[:, i] += np.random.normal(0, sigma, data.shape[0])

		return data, _labels


	def fit(self, data_in):
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
		self.dtype = data_in.dtype

		data = flatten_struc_array(data_in)

		data, labels = self.process_hints(data)

		self.fits_to_run = []
		self._branch_fit(data, labels[:])

		# for fit_info in self.fits_to_run:
			# self._fit_count += fit_info[0].shape[0]

		self._fit_count = len(self.fits_to_run)

		while True:
			self.progress_bar()
			try:
				d, labels, ins_cmd = self.fits_to_run.pop()
			except IndexError:
				break
			self.single_fit(d, labels=labels, insert_cmd=ins_cmd)

		if self.config['verbose'] > 0:
			sys.stderr.write("\n")


	def sample_dist(self, dist, hint, n):
		""" """
		if dist == 'uniform':
			low, high = hint[1:3]
			self.logger.info("random sampling from uniform distribution %s %s %s", low, high, n)
			x = np.random.uniform(low, high, n)
			return x

		raise SynException("don't know how to sample from dist: %s", hint)

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

		count_total = 0

		syndata = np.zeros(n, dtype=self.dtype)

		while count_total < n:
			for fit in self.fit_results:

				# number of samples to draw
				batch = int(np.round(fit.count * n * 1. / total_count))

				start = count_total
				count = 0
				while count < batch:
					sample = fit.gmm.sample(batch - count)

					if fit.pca is not None:
						sample = fit.pca.inverse_transform(sample)

					sample = sample * fit.sigma + fit.mu

					for func, i in fit.transform:
						sample[:, i] = func(sample[:, i])

					if 'truncate' in self.hints:
						for name, low, high in self.hints['truncate']:
							try:
								i = fit.labels.index(name)
							except ValueError:
								continue
							select = np.where((sample[:, i] > low) & (sample[:, i] < high))
							n_0 = len(sample)
							sample = sample[select]
							self.logger.debug("truncating %s %s %s: %i -> %i", name, low, high, n_0, len(sample))

					insert_count = struc_array_insert(syndata, sample, fit.labels, count_total)

					if fit.insert_cmd is not None:
						for column_name, value in fit.insert_cmd:
							# value_arr = np.ones((insert_count, 1)) * value
							# struc_array_insert(syndata, value_arr, [column_name], count_total)
							syndata[column_name][count_total:count_total+insert_count] = value

					count += insert_count
					count_total += insert_count

					if count_total >= n:
						break

				if count_total > n:
					break

		assert count_total == n

		for instruction, column_name, hint in self.other_dists[::-1]:
			syndata[column_name] = self.sample_dist(instruction, hint, n)

		return syndata


class SynException(Exception):
	pass
