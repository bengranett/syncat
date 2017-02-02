import sys,os
import numpy as N
import logging

import cPickle as pickle

from sklearn import mixture

from pypelid.utils.misc import increment_path

class SimpleGMM:
	def __init__(self, components=[]):
		""" Gaussian mixture model class """

		self.means_ = []
		self.covars_ = []
		self.weights_ = []

		for a,b,c in components:

			b = N.array(b)
			if b.size==len(b):
				b = N.diag(b)

			self.means_.append(a)
			self.covars_.append(b)
			self.weights_.append(c)

		self.dim = len(self.means_[0])

	def add(self, mu=0, cov=1, w=1):
		""" Add a Gaussian component """
		if self.dim is not None:
			assert(len(mu)==self.dim)
		else:
			self.dim = len(mu)

		cov = N.array(cov)
		if cov.size==len(cov):
			cov = N.diag(cov)

		self.means_.append(mu)
		self.covars_.append(cov)
		self.weights_.append(w)

	def sample(self, n, random_state=None):
		""" Sample from the GMM """
		N.random.seed(random_state)

		norm = N.sum(self.weights_)
		if norm==0:
			print "weights are all 0!"
			return
		w = N.array(self.weights_)*1./norm

		ncomp = len(w)

		ii = N.random.choice(N.arange(ncomp), n, w)

		m = N.bincount(ii)

		samp = []
		for i in range(ncomp):
			s = N.random.multivariate_normal(self.means_[i],self.covars_[i],m[i])
			samp.append(s)

		samp = N.vstack(samp)

		order = N.random.uniform(0,1,len(samp)).argsort()
		samp = samp[order]

		return samp


class Syn:
	""" """
	logger = logging.getLogger(__name__)

	def __init__(self, cachefile=None):
		""" Build synthetic datasets. """
		self.fit_results = []

		self.cachefile = cachefile

		self.load(self.cachefile)

	def load(self, path):
		""" Load a cache file from a previous run. """
		if path is not None and os.path.exists(path):
			self.fit_results = pickle.load(file(path))

	def save(self, path=None):
		""" Save a cache file. """
		if path is None:
			path = self.cachefile

		dir = os.path.dirname(path)
		if not os.path.exists(dir):
			os.mkdir(dir)

		path = increment_path(path)

		if os.path.exists(path):
			self.logger.warning("File exists: %s.  Will not overwrite.", path)
		else:
			pickle.dump(self.fit_results, file(path,'w'))
			self.logger.info("wrote %s", path)

	def set(self, comp):
		""" Build a GMM class with these components. """
		G = SimpleGMM(comp)
		dim = G.dim
		mu = N.zeros(dim)
		sig = N.ones(dim)
		self.fit_results = [(mu,sig,G,-1,None)]

	def _fit(self, data, k=30, loops=10, labels=None):
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
		labels: list of column names

		Output
		------
		None

		"""
		mu = N.mean(data, axis=0)
		sig = N.std(data, axis=0)

		# print mu,sig
		data_w = (data - mu)/sig

		best = None
		best_aic = None
		for i in range(loops):
			G = mixture.GMM(n_components=k, covariance_type='full')
			G.fit(data_w)
			if not G.converged_: continue
			aic = G.aic(data_w)

			if best_aic is None:
				self.logger.debug("loop %i, AIC %f", i, G.aic(data_w))
			else:
				self.logger.debug("loop %i, AIC %f (best so far %f)", i, G.aic(data_w), best_aic)

			if best_aic is None or aic < best_aic:
				best = G
				best_aic = aic

		if best == None:
			self.logger.error("dead! no good gmm fit was found!")
			exit(-1)

		G = best

		params = (mu, sig, G, best_aic, labels)

		return params


	def fit(self, data, batch_size=10000, k=30, loops=2, labels=None):
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
		labels: list of column names

		Output
		------
		None

		"""
		dim, n = data.shape
		assert dim < n

		try:
			assert dim == self.dim
		except AttributeError:
			self.dim = dim

		# adjust batch size so we have equal sized batches
		nbatch = int(np.floor(n * 1. / batch_size))
		batch_size = n // nbatch

		self.logger.debug("data size %i", n)
		self.logger.debug("Batch size %i", batch_size)

		bins = np.arange(0, n, batch_size)
		self.logger.debug("bins: %s", bins)

		for i in bins:
			j = i + batch_size
			sub = data[:, i:j]

			params = self._fit(sub, k, loops, labels)

			self.fit_results.append(params)


	def sample(self, n=1e5, random_state=None, save=None):
		""" Draw samples from the model. """
		if save is not None:
			save = increment_path(save)

		nfits = len(self.fit_results)
		batch = int(n * 1. / nfits)

		data_out = []
		for fit in self.fit_results:
			mu, sig, G, best_aic, labels = fit

			syndata = G.sample(batch)

			syndata = syndata * sig + mu

			data_out.append(syndata)

		data_out = np.hstack(data_out)

		return data_out
