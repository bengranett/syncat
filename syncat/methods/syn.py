import sys, os
import numpy as np
import logging
import time

import cPickle as pickle
from sklearn import mixture, decomposition

from pypeline import pype, add_param

from syncat.misc import flatten_struc_array, struc_array_insert




class FitResults(object):
	""" Data structure to store parameters of the fit. """
	logger = logging.getLogger(__name__)

	def __init__(self, **kwargs):
		""" """
		self.norm = None
		self.__dict__.update(kwargs)

	def sample_me(self, n):
		""" """
		sample, sample_labels = self.gmm.sample(n)

		if self.pca is not None:
			sample = self.pca.inverse_transform(sample)

		sample = sample * self.sigma + self.mu

		for func, i in self.transform:
			sample[:, i] = func(sample[:, i])

		return sample

	def determine_pdf_norm(self, n=1e6):
		""" """
		if self.norm is not None:
			return self.norm

		sample, sample_labels = self.gmm.sample(n)

		p = np.exp(self.gmm.score_samples(sample))

		self.norm = 1. / p.max()

		return self.norm

	def pdf(self, points):
		""" evaluate pdf at a point """
		#if self.norm is None:
	#		self.determine_pdf_norm()

		points_t = points.copy()

		for func, i in self.invtransform:
			points_t[:, i] = func(points_t[:, i])

		points_t = (points_t - self.mu) / self.sigma

		return np.exp(self.gmm.score_samples(points_t))


@add_param('ncomponents', metavar='n', default=10, type=int, help="Number of Gaussian components to mix")
@add_param('batch_size', metavar='n', default=10000, type=int, help="Fit in batches of this size", hidden=True)
@add_param('nloops', metavar='n', default=1, type=int, help="Number of times to repeat fit and take the best one")
@add_param('apply_pca', default=False, action='store_true', help="Preprocess with PCA", hidden=True)
@add_param('covariance_mode', metavar='mode', default='diag', choices=['diag', 'full', 'tied', 'spherical'],
	help="Mode for modelling the Gaussian components.", hidden=True)
@add_param('min_fit_size',metavar='n',  default=50, type=int, help="Minimimum size of dataset to do Gaussian fit", hidden=True)
@add_param('discreteness_count', metavar='n', default=20, type=int, help="If count of unique values is less than this, array is discrete", hidden=True)
@add_param('special_values', metavar='x', default=[0, -99, float('nan')], nargs='*', type=float, help='values that will be treated as discrete.')
@add_param('log_crit', metavar='x', default=1, help='will try a log transform if the dynamic range is greater than this and all positive.', hidden=True)
@add_param('tol_const', metavar='x', default=1e5, type=float, help='tolerance constant for checking discrete values', hidden=True)
@add_param('gmm_max_iter', metavar='n', default=100, type=int, help='scikit-learn gaussian mixture model max_iter parameter', hidden=True)
@add_param('gmm_ninit', metavar='n', default=1, type=int, help='scikit-learn gaussian mixture model n init parameter', hidden=True)
@add_param('verbose', alias='v', default=0, type=int, help='verbosity level')
class Syn(pype):
	""" """
	def __init__(self, loadfile=None, labels=None, hints={}, config={}, **kwargs):
		""" Build synthetic datasets. """
		self._parse_config(config, **kwargs)
		self._setup_logging()

		self.labels = labels
		self.loadfile = loadfile

		self.fit_results = []

		self.hints = hints

		self.load(self.loadfile)

	def load(self, path):
		""" Load a catalogue model file from a previous run. """
		if path is not None and os.path.exists(path):
			self.undump(pickle.load(file(path)))
			self.logger.info("Loaded %s, got %i fit results.", path, len(self.fit_results))

	def dump(self):
		return self.labels, self.fit_results, self.other_dists, self.hints, self.dtype, self.lookup

	def undump(self, dump):
		""" """
		self.labels, self.fit_results, self.other_dists, self.hints, self.dtype, self.lookup = dump

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
		pickle.dump(self.dump(), file(path, 'w'))
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
			G = mixture.GaussianMixture(n_components=k, covariance_type=self.config['covariance_mode'], 
						    max_iter=self.config['gmm_max_iter'], n_init=self.config['gmm_ninit'])
			G.fit(data_w)
			if not G.converged_:
				continue

			if loops > 1:
				aic = G.aic(data_w)

			if best_aic is None or aic < best_aic:
				best = G
				best_aic = aic

		if best is None:
			np.save('datadump.npy', data_w)
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
		invlogtransform = []
		for i in range(data.shape[1]):
			low, high = data[:, i].min(), data[:, i].max()
			try:
				assert high > low
			except:
				print i,labels[i],low,high
				raise
			if low > 0 and high / low > self.config['log_crit']:
				pass
				# self.logger.info("\nlog transform %s %s %s", labels[i], low, high)
				#data[:, i] = np.log(data[:, i])

				#logtransform.append((np.exp, i))
				#invlogtransform.append((np.log, i))

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
			fit.invtransform = invlogtransform

			fit.hash = hash(tuple(fit.insert_cmd))

			self.fit_results.append(fit)

	def _branch_fit(self, data, labels, column=0, insert=[]):
		""" This routine is called recursively.  

		1. If column i is the last column add GMM fit job
		1. Identify discrete values in column i
		2. For each discrete value do a selection on the data table, increment the column
		   and call _branch_fit again
		"""
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
			# otherwise use a list of default special values eg 0, -99, nan,...
			discrete = False
			special_values = self.config['special_values']

		unmatched = np.ones(n, dtype=bool)

		# compute absolute tolerance for equality
		mu = np.abs(data[:, column]).mean()
		tol = mu / self.config['tol_const']
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
					try:
						i = _labels.index(key)
					except ValueError:
						continue
					_labels.pop(i)
					data = np.delete(data, i, axis=1)
					self.other_dists.append((instruction, key, hint))
			elif instruction == 'smooth':
				for hint in hints:
					key = hint[0]
					sigma = float(hint[1])
					try:
						i = _labels.index(key)
					except ValueError:
						continue
					self.logger.info("Smoothing %s sigma=%f", key, sigma)
					data[:, i] += np.random.normal(0, sigma, data.shape[0])

		return data, _labels

	def fit(self, data_in, dtype=None):
		""" Fit the data with a Gaussian mixture model.
		The data are partitioned to handle discrete columns and special values.

		Parameters
		----------
		data : ndarray
			catalogue data to fit. shape should be (nobj, dimension)

		Output
		------
		None
		"""
		if dtype is None:
			self.dtype = data_in.dtype
		else:
			self.dtype = dtype

		data = flatten_struc_array(data_in, type='d')

		data, labels = self.process_hints(data)

		self.discrete_values = {}
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

		self.lookup = {}
		for fit in self.fit_results:
			if fit.hash not in self.lookup:
				self.lookup[fit.hash] = []
			self.lookup[fit.hash].append(fit)

		if self.config['verbose'] > 0:
			sys.stderr.write("\n")

	def pdf(self, point):
		""" Evaluate the probability distribution function at a point

		Parameters
		----------

		Returns
		-------
		"""
		for fit in self.fit_results:

			point_t = point.copy()

			print fit.insert_cmd
			continue

			for func, i in fit.invlogtransform:
				point_t[i] = func(point[i])

			point_t = (point_t - fit.mu) / fit.sigma


	def sample_dist(self, dist, hint, n):
		""" """
		if dist == 'uniform':
			low, high = hint[1:3]
			self.logger.debug("random sampling from uniform distribution %s %s %s", low, high, n)
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

					sample = fit.sample_me(batch - count)

					if 'truncate' in self.hints:
						for name, low, high in self.hints['truncate']:
							try:
								i = fit.labels.index(name)
							except ValueError:
								continue
							select = np.where((sample[:, i] > low) & (sample[:, i] < high))

							n_0 = len(sample)
							sample = sample[select]
							# self.logger.debug("truncating %s %s %s: %i -> %i", name, low, high, n_0, len(sample))

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
