""" Some general tools.

	https://docs.python.org/2/distutils/introduction.html

	Markovic, 05/03/2015
"""

import numpy as np
import logging

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

def append_dtypes(dtype, names, dtype_lookup, translate=None):
	""" Append a dtype given a list of names
	
	Parameters
	----------
	dtype : numpy.dtype
	names : tuple
	dtype_lookup : dict
	transalte : func

	Returns
	-------
	numpy.dtype
	"""
	if translate is None:
		translate = lambda x: x
		
	for name in names:
		name_out = translate(name)
		if name_out in dtype.names:
			continue
		try:
			dtype = concatenate_dtypes([dtype, np.dtype([(name_out, dtype_lookup[name])])])
		except KeyError:
			pass
	return dtype

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

def flatten_struc_array(arr, type=None):
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

	if type is None:
		type = arr.dtype[0]

	ncol = len(arr.dtype)
	nrow = len(arr)
	arr_out = np.zeros((nrow, ncol), dtype=type)

	for i, name in enumerate(arr.dtype.names):
		arr_out[:,i] = arr[name]

	return arr_out

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

def struc_array_columns(arr, columns):
	""" Take columns from a numpy structured array"""
	out = []
	for column in columns:
		out.append(arr[column])
	return np.transpose(out)

def insert_column(columns, in_array, out_array, translate=None, columns_added=[]):
	""" """
	if translate is None:
		translate = lambda x: x

	new_columns = []

	for column in columns:
		if column in out_array.dtype.names:
			column_out = translate(column)
			if column_out in columns_added:
				continue
			if column_out in out_array.dtype.names:
				out_array[column_out] = in_array[column]
				new_columns.append(column_out)
	return new_columns

def remove_columns(table, skip_columns):
	""" """
	properties = []

	for name in table.columns:
		hit = False
		for skip in skip_columns:
			if skip.lower() == name.lower():
				hit = True
				logging.info("ignoring column '%s' because it matches the string '%s'.", name, skip)
				break

		if not hit:
			properties.append(name)

	return table[properties]