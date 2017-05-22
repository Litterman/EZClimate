from __future__ import division, print_function
import numpy as np
from scipy.optimize import fmin

class CoordinateDescent(object):
	"""Coordinate descent optimization algorithm for the EZ-Climate model.

	Parameters
	----------
	utility : `Utility` object
		object of utility class
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the utility increase 
	iterations : int 
		maximum number of iterations

	Attributes
	----------
	utility : `Utility` object
		object of utility class
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the utility increase
	iterations : int 
		maximum number of iterations

	"""
	def __init__(self, utility, var_nums, accuracy=1e-4, iterations=100):
		self.u = utility
		self.var_nums = var_nums
		self.accuracy = accuracy
		self.iterations = iterations
	
	def _min_func(self, x, m, i):
  		m_copy = m.copy()
   		m_copy[i] = x
   		return -self.u.utility(m_copy)[0]

	def _minimize_node(self, node, m):
		return fmin(self._min_func, x0=m[node], args=(m, node), disp=False)

	def run(self, m):
		"""Run the coordinate descent iterations.

		Parameters
		----------
		m : initial point

		Returns
		-------
		tuple
			best mitigation point and the utility of the best mitigation point

		Note
		----
		Uses the :mod:`~scipy` package.

		"""
		num_decision_nodes = m.shape[0]
		x_hist = []
		u_hist = []
		nodes = range(self.var_nums)
		x_hist.append(m.copy())
		u_hist.append(self.u.utility(m)[0])
		print("----------------Coordinate Descent Starting----------------")
		print("Starting Utility: {}".format(u_hist[0]))
		for i in range(self.iterations):
			print("-- Iteration {} --".format(i+1))
			node_iteration = np.random.choice(nodes, replace=False, size=len(nodes))
			for node in node_iteration:
				m[node] = max(0.0, self._minimize_node(node, m))
			x_hist.append(m.copy())
			u_hist.append(self.u.utility(m)[0])
			print("Current Utility: {}".format(u_hist[i+1]))
			if np.abs(u_hist[i+1] - u_hist[i]) < self.accuracy:
				break
		return x_hist[-1], u_hist[-1]