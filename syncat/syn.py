import sys,os
import numpy as N

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

	def __init__(self, logz = False, cachefile=None):
		""" Build synthetic datasets. """
		self.cachefile = cachefile

		self.logz = logz
		self.load(self.cachefile)

	def load(self, path):
		""" Load a cache file from a previous run. """
		if path is not None and os.path.exists(path):
			self.params = pickle.load(file(path))
			self.dim = len(self.params[0])

	def save(self, path=None):
		""" Save a cache file. """
		if path is None: path = self.cachefile

		dir = os.path.dirname(path)
		if not os.path.exists(dir): os.mkdir(dir)

		path = increment_path(path)

		if os.path.exists(path):
			print "File exists: %s.  Will not overwrite."%path
		else:
			print "> wrote",path
			pickle.dump(self.params, file(path,'w'))

	def set(self, comp):
		""" Build a GMM class with these components. """
		print comp
		G = SimpleGMM(comp)
		dim = G.dim
		mu = N.zeros(dim)
		sig = N.ones(dim)
		self.dim = dim
		self.params = (mu,sig,G,-1,None)
		print "dim=",dim

	def fit(self, data, k=30, loops=10, labels=None):
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
		print "running gausian fit with %i components"%k

		# log z
		if self.logz:
			z = data[:,0]
			assert(N.all(z>0))
			datat = data.copy()
			datat[:,0] = N.log(z)
		else:
			datat = data

		# whiten data
		mu = N.mean(datat, axis=0)
		sig = N.std(datat, axis=0)
		# print mu,sig
		data_w = (datat - mu)/sig

		best = None
		best_aic = None
		for i in range(loops):
			G = mixture.GMM(n_components=k, covariance_type='full')
			G.fit(data_w)
			if not G.converged_: continue
			aic = G.aic(data_w)

			if best_aic is None:
				print i, "AIC %f"%G.aic(data_w)
			else:
				print i, "AIC %f (best so far %f)"%(G.aic(data_w),best_aic)

			if best_aic is None or aic < best_aic:
				best = G
				best_aic = aic

		if best == None:
			print "dead! no good gmm fit was found!"
			exit(-1)

		G = best

		self.dim = len(mu)
		self.params = (mu, sig, G, best_aic, labels)


	def sample(self, n=1e5, random_state=None, save=None):
		""" Draw samples from the model. """
		if save is not None:
			save = increment_path(save)

		mu, sig, G, best_aic, labels = self.params

		print "generating data"
		syndata = G.sample(int(n),random_state=random_state)

		syndata = syndata*sig + mu

		if self.logz:
			syndata[:,0] = N.exp(syndata[:,0])

		if save is not None:
			N.save(save, syndata)

		print "done",syndata.shape
		return syndata
