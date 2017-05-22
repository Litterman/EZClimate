from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

class BaseStorageTree(object):
	"""Abstract storage class for the EZ-Climate model.

	Parameters
	----------
	decision_times : ndarray or list
		array of years from start where decisions about mitigation levels are done
	
	Attributes
	----------
	decision_times : ndarray 
		array of years from start where decisions about mitigation levels are done
	information_times : ndarray 
		array of years where new information is given to the agent in the model
	periods : ndarray 
		periods in the tree
	tree : dict
		dictionary where keys are `periods` and values are nodes in period

	"""
	__metaclass__ = ABCMeta

	def __init__(self, decision_times):
		self.decision_times = decision_times
		if isinstance(decision_times, list):
			self.decision_times = np.array(decision_times)
		self.information_times = self.decision_times[:-2]
		self.periods = None
		self.tree = None

	def __len__(self):
		return len(self.tree)

	def __getitem__(self, key):
		if isinstance(key, int) or isinstance(key, float):
			return self.tree.__getitem__(key).copy()
		else:
			raise TypeError('Index must be int, not {}'.format(type(key).__name__))

	def _init_tree(self):
		self.tree = dict.fromkeys(self.periods)
		i = 0
		for key in self.periods:
			self.tree[key] = np.zeros(2**i)
			if key in self.information_times:
				i += 1
	@property
	def last(self):
		"""ndarray: last period's array."""
		return self.tree[self.decision_times[-1]]

	@property
	def last_period(self):
		"""int: index of last period."""
		return self.decision_times[-1]

	@property
	def nodes(self):
		"""int: number of nodes in the tree."""
		n = 0
		for array in self.tree.values():
			n += len(array)
		return n

	@abstractmethod
	def get_next_period_array(self, period):
		"""Return the array of the next period from `periods`."""
		pass

	def set_value(self, period, values):
		"""If period is in periods, set the value of element to `values` (ndarray)."""
		if period not in self.periods:
			raise ValueError("Not a valid period")
		if isinstance(values, list):
			values = np.array(values)
		if self.tree[period].shape != values.shape:
			raise ValueError("shapes {} and {} not aligned".format(self.tree[period].shape, values.shape))
		self.tree[period] = values

	def is_decision_period(self, time_period):
		"""Checks if time_period is a decision time for mitigation, where
		time_period is the number of years since start.

		Parameters
		----------
		time_period : int
			time since the start year of the model

		Returns
		-------
		bool
			True if time_period also is a decision time, else False

		"""
		return time_period in self.decision_times

	def is_real_decision_period(self, time_period):
		"""Checks if time_period is a decision time besides the last period, where
		time_period is the number of years since start.
		
		Parameters
		----------
		time_period : int
			time since the start year of the model

		Returns
		-------
		bool
			True if time_period also is a real decision time, else False

		"""
		return time_period in self.decision_times[:-1]

	def is_information_period(self, time_period):
		"""Checks if time_period is a information time for fragility, where
		time_period is the number of years since start.

		Parameters
		----------
		time_period : int
			time since the start year of the model

		Returns
		-------
		bool
			True if time_period also is an information time, else False

		"""
		return time_period in self.information_times

	def write_tree(self, file_name, header, delimiter=";"):
		"""Save values in `tree` as a tree into file  `file_name` in the 
		'data' directory in the current working directory. If there is no 'data' 
		directory, one is created. 
 
		Parameters
		----------
		file_name : str
			name of saved file
		header : str
			first row of file
		delimiter : str, optional
			delimiter in file

		"""
		from tools import find_path
		import csv
		
		real_times = self.decision_times[:-1]
		size = len(self.tree[real_times[-1]])
		output_lst = []
		prev_k = size

		for t in real_times:
			temp_lst = [""]*(size*2)
			k = int(size/len(self.tree[t]))
			temp_lst[k::prev_k] = self.tree[t].tolist()
			output_lst.append(temp_lst)
			prev_k = k

		write_lst = zip(*output_lst)
		d = find_path(file_name)
		with open(d, 'wb') as f:
			writer = csv.writer(f, delimiter=delimiter)
			writer.writerow([header])
			for row in write_lst:
				writer.writerow(row)
	
	def write_columns(self, file_name, header, start_year=2015, delimiter=";"):
		"""Save values in `tree` as columns into file  `file_name` in the 
		'data' directory in the current working directory. If there is no 'data' 
		directory, one is created. 
			
		+------------+------------+-----------+
		|    Year    |    Node 	  |  header   |
		+============+============+===========+
		| start_year |     0	  |   val0    |
		+------------+------------+-----------+
		|     ..     |	   .. 	  |    ..     |
		+------------+------------+-----------+
		
		Parameters
		----------
		file_name : str
			name of saved file
		header : str
			description of values in tree
		start_year : int, optional
			start year of analysis
		delimiter : str, optional
			delimiter in file

		"""
		from tools import write_columns_csv, file_exists
		if file_exists(file_name):
			self.write_columns_existing(file_name, header)
		else:
			real_times = self.decision_times[:-1]
			years = []
			nodes = []
			output_lst = []
			k = 0
			for t in real_times:
				for n in range(len(self.tree[t])):
					years.append(t+start_year)
					nodes.append(k)
					output_lst.append(self.tree[t][n])
					k += 1
			write_columns_csv(lst=[output_lst], file_name=file_name, header=["Year", "Node", header], 
							  index=[years, nodes], delimiter=delimiter)

	def write_columns_existing(self, file_name, header, delimiter=";"):
		"""Save values in `tree` as columns into file  `file_name` in the 
		'data' directory in the current working directory, when `file_name` already exists. 
		If there is no 'data' directory, one is created. 
			
		+------------+------------+-----------------+------------------+
		|    Year    |    Node    |  other_header   |      header      |
		+============+============+=================+==================+
		| start_year |     0      |   other_val0    | 	    val0       |
		+------------+------------+-----------------+------------------+
		|     ..     |     ..     |       ..        |        ..        |
		+------------+------------+-----------------+------------------+

		Parameters
		----------
		file_name : str
			name of saved file
		header : str
			description of values in tree
		start_year : int, optional
			start year of analysis
		delimiter : str, optional
			delimiter in file

		"""
		from tools import write_columns_to_existing
		output_lst = []
		for t in self.decision_times[:-1]:
			output_lst.extend(self.tree[t])
		write_columns_to_existing(lst=output_lst, file_name=file_name, header=header)


class SmallStorageTree(BaseStorageTree):
	"""Storage tree class for the EZ-Climate model. No storage in nodes between 
	periods in `decision_times`.

	Parameters
	----------
	decision_times : ndarray or list
		array of years from start where decisions about mitigation levels are done

	Attributes
	----------
	decision_times : ndarray 
		array of years from start where decisions about mitigation levels are done
	information_times : ndarray 
		array of years where new information is given to the agent in the model
	periods : ndarray 
		periods in the tree
	tree : dict
		dictionary where keys are `periods` and values are nodes in period

	"""
	def __init__(self, decision_times):
		super(SmallStorageTree, self).__init__(decision_times)
		self.periods = self.decision_times
		self._init_tree()

	def get_next_period_array(self, period):
		"""Returns the array of the next decision period.

		Parameters
		----------
		period : int
			period

		Examples
		--------
		>>> sst = SmallStorageTree([0, 15, 45, 85, 185, 285, 385])
		>>> sst.get_next_period_array(0)
		array([0., 0.])
		>>> sst.get_next_period_array(15)
		array([ 0.,  0.,  0.,  0.])

		Raises
		------
		IndexError
			If `period` is not in real decision times

		"""
		if self.is_real_decision_period(period):
			index = self.decision_times[np.where(self.decision_times==period)[0]+1][0]
			return self.tree[index].copy()
		raise IndexError("Given period is not in real decision times")

	def index_below(self, period):
		"""Returns the key of the previous decision period.

		Parameters
		----------
		period : int
			period

		Examples
		--------
		>>> sst = SmallStorageTree([0, 15, 45, 85, 185, 285, 385])
		>>> sst.index_below(15)
		0

		Raises
		------
		IndexError
			If `period` is not in decision times or first element in decision times

		"""
		if period in self.decision_times[1:]:
			period = self.decision_times[np.where(self.decision_times==period)[0]-1]
			return period[0]
		raise IndexError("Period not in decision times or first period")

class BigStorageTree(BaseStorageTree):
	"""Storage tree class for the EZ-Climate model. Storage in nodes between 
	periods in `decision_times`. 

	Parameters
	----------
	subintervals_len : float
		years between periods in tree
	decision_times : ndarray or list
		array of years from start where decisions about mitigation levels are done

	Attributes
	----------
	decision_times : ndarray 
		array of years from start where decisions about mitigation levels are done
	information_times : ndarray 
		array of years where new information is given to the agent in the model
	periods : ndarray 
		periods in the tree
	tree : dict
		dictionary where keys are `periods` and values are nodes in period
	subintervals_len : float
		years between periods in tree

	"""

	def __init__(self, subinterval_len, decision_times):
		super(BigStorageTree, self).__init__(decision_times)
		self.subinterval_len = subinterval_len
		self.periods = np.arange(0, self.decision_times[-1]+self.subinterval_len,
							 self.subinterval_len)
		self._init_tree()

	@property
	def first_period_intervals(self):
		"""ndarray: the number of subintervals in the first period."""
		return int((self.decision_times[1] - self.decision_times[0]) / self.subinterval_len)

	def get_next_period_array(self, period):
		"""Returns the array of the next period.

		Parameters
		----------
		period : int
			period

		Examples
		--------
		>>> bst = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
		>>> sst.get_next_period_array(0)
		array([0., 0.])
		>>> sst.get_next_period_array(10)
		array([ 0.,  0., 0., 0.])

		Raises
		------
		IndexError
			If `period` is not a valid period or too large

		"""
		if period + self.subinterval_len <= self.decision_times[-1]:
			return self.tree[period+self.subinterval_len].copy()
		raise IndexError("Period is not a valid period or too large")

	def between_decision_times(self, period):
		"""Check which decision time the period is between and returns
		the index of the lower decision time. 

		Parameters
		----------
		period : int
			period

		Returns
		-------
		int
			index

		Examples
		--------
		>>> bst = BigStorageTree(5, [0, 15, 45, 85, 185, 285, 385])
		>>> bst.between_decision_times(5)
		0
		>>> bst.between_decision_times(15)
		1

		"""
		if period == 0:
			return 0
		for i in range(len(self.information_times)):
			if self.decision_times[i] <= period and period < self.decision_times[i+1]:
				return i
		return i+1

	def decision_interval(self, period):
		"""Check which interval the period is between.
		
		Parameters
		----------
		period : int
			period

		Returns
		-------
		int
			index

		Examples
		--------
		>>> bst = BigStorageTree(5, [0, 15, 45, 85, 185, 285, 385])
		>>> bst.decision_interval(5)
		1
		>>> bst.between_decision_times(15)
		1
		>>> bst.between_decision_times(20)
		2

		"""
		if period == 0:
			return 0
		for i in range(1, len(self.decision_times)):
			if self.decision_times[i-1] < period and period <= self.decision_times[i]:
				return i
		return i

