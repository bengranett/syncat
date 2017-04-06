""" Some general tools.

 	https://docs.python.org/2/distutils/introduction.html

 	Markovic, 05/03/2015
"""

import numpy as np

def ensurelist(supposed_list):
	if isinstance(supposed_list,list):
		return supposed_list
	else:
		return [supposed_list]

def dict_to_structured_array(data_dict):
	""" Convert a dictionary of numpy arrays to a structured array. 

	Inputs
	------
	data_dict - dictionary of numpy arrays

	Outputs
	-------
	structured array
	"""
	lengths = []
	dtypes = []
	for name, arr in data_dict.items():
		lengths.append(arr.shape[0])
		dim=1
		if len(arr.shape)>1:
			dim = arr.shape[1:]


		dtypes.append((name, arr.dtype, dim))

	lengths = np.array(lengths)
	if not np.all(lengths == lengths[0]):
		raise ValueError("Not all arrays in the dictionary have the same length.")

	# initialize the empty structured array
	struc_array = np.zeros(lengths[0], dtype=dtypes)

	# load the data
	for name, arr in data_dict.items():
		struc_array[name] = arr

	return struc_array

def concatenate_dtypes(dtypes):
	""" Combine a list of dtypes.
	This may be used to add columns to a dtype.

	Parameters
	----------
	dtypes : sequence
		sequence of dtypes

	Results
	---------
	dtype : numpy.dtype
		the combined dtype object
	"""
	dtype_out = []
	for dtype in dtypes:
		for name in dtype.names:
			dtype_out.append((name, dtype[name]))
	return np.dtype(dtype_out)

def flatten_struc_array(arr):
	"""Convert a structured array to a numpy ndarray without column names.

	The columns of the input array must be 1-dimensional and all of the same type.

	Parameters
	----------
	arr : numpy ndarray
		Input array to convert

	Returns
	-------
	numpy ndarray
	"""
	if len(arr.dtype)==0:
		return arr

	t = arr.dtype[0]

	ncol = len(arr.dtype)
	nrow = len(arr)
	arr_out = np.zeros((nrow, ncol), dtype=t)

	for i, name in enumerate(arr.dtype.names):
		arr_out[:,i] = arr[name]

	return arr_out
