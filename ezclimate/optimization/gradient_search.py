from __future__ import division, print_function
import numpy as np
import multiprocessing
from ezclimate.tools import _pickle_method, _unpickle_method
try:
    import copy_reg
except:
    import copyreg as copy_reg
import types

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class GradientSearch(object) :
	"""Gradient search optimization algorithm for the EZ-Climate model.

	Parameters
	----------
	utility : `Utility` object
		object of utility class
	learning_rate : float
		starting learning rate of gradient descent
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the gradient descent
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed
	scale_alpha : ndarray, optional
		array to scale the learning rate

	Attributes
	----------
	utility : `Utility` object
		object of utility class
	learning_rate : float
		starting learning rate of gradient descent
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the gradient descent
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed
	scale_alpha : ndarray, optional
		array to scale the learning rate

	"""

	def __init__(self, utility, var_nums, accuracy=1e-06, iterations=100, fixed_values=None, 
		        fixed_indicies=None, print_progress=False, scale_alpha=None):
		self.u = utility
		self.var_nums = var_nums
		self.accuracy = accuracy
		self.iterations = iterations
		self.fixed_values  = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress
		self.scale_alpha = scale_alpha
		if scale_alpha is None:
			self.scale_alpha = np.exp(np.linspace(0.0, 3.0, var_nums))

	def _partial_grad(self, i):
		"""Calculate the ith element of the gradient vector."""
		m_copy = self.m.copy()
		m_copy[i] = m_copy[i] - self.delta if (m_copy[i] - self.delta)>=0 else 0.0
		minus_utility = self.u.utility(m_copy)
		m_copy[i] += 2*self.delta
		plus_utility = self.u.utility(m_copy)
		grad = (plus_utility-minus_utility) / (2*self.delta)
		return grad, i

	def numerical_gradient(self, m, delta=1e-08, fixed_indicies=None):
		"""Calculate utility gradient numerically.

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		delta : float, optional
			change in mitigation 
		fixed_indicies : ndarray or list, optional
			indicies of gradient that should not be calculated

		Returns
		-------
		ndarray
			gradient

		"""
		self.delta = delta
		self.m = m
		if fixed_indicies is None:
			fixed_indicies = []
		grad = np.zeros(len(m))
		if not isinstance(m, np.ndarray):
			self.m = np.array(m)
		pool = multiprocessing.Pool()
		indicies = np.delete(range(len(m)), fixed_indicies)
		res = pool.map(self._partial_grad, indicies)
		for g, i in res:
			grad[i] = g
		pool.close()
		pool.join()
		del self.m
		del self.delta
		return grad

	def _accelerate_scale(self, accelerator, prev_grad, grad):
		sign_vector = np.sign(prev_grad * grad)
		scale_vector = np.ones(self.var_nums) * ( 1 + 0.10)
		accelerator[sign_vector <= 0] = 1
		accelerator *= scale_vector
		return accelerator


	def gradient_descent(self, initial_point, return_last=False):
		"""Gradient descent algorithm. The `initial_point` is updated using the 
		Adam algorithm. Adam uses the history of the gradient to compute individual 
		step sizes for each element in the mitigation vector. The vector of step 
		sizes are calculated using estimates of the first and second moments of 
		the gradient.

		Parameters
		----------
		initial_point : ndarray
			initial guess of the mitigation
		return_last : bool, optional
			if True the function returns the last point, else the point 
				with highest utility

		Returns
		-------
		tuple
			(best point, best utility)
		
		"""
		num_decision_nodes = initial_point.shape[0]
		x_hist = np.zeros((self.iterations+1, num_decision_nodes))
		u_hist = np.zeros(self.iterations+1)
		u_hist[0] = self.u.utility(initial_point)
		x_hist[0] = initial_point
		
		beta1, beta2 = 0.90, 0.90
		eta = 0.0015
		eps = 1e-3
		m_t, v_t = 0, 0

		prev_grad = 0.0
		accelerator = np.ones(self.var_nums)

		for i in range(self.iterations):
			grad = self.numerical_gradient(x_hist[i], fixed_indicies=self.fixed_indicies)
			m_t = beta1*m_t + (1-beta1)*grad
			v_t = beta2*v_t + (1-beta2)*np.power(grad, 2) 
			m_hat = m_t / (1-beta1**(i+1))
			v_hat = v_t / (1-beta2**(i+1))
			if i != 0:
				accelerator = self._accelerate_scale(accelerator, prev_grad, grad)
			
			new_x = x_hist[i] + ((eta*m_hat)/(np.square(v_hat)+eps)) * accelerator
			new_x[new_x < 0] = 0.0

			if self.fixed_values is not None:
				new_x[self.fixed_indicies] = self.fixed_values

			x_hist[i+1] = new_x
			u_hist[i+1] = self.u.utility(new_x)[0]
			prev_grad = grad.copy()

			if self.print_progress:
				print("-- Iteration {} -- \n Current Utility: {}".format(i+1, u_hist[i+1]))
				print(new_x)

		if return_last:
			return x_hist[i+1], u_hist[i+1]
		best_index = np.argmax(u_hist)
		return x_hist[best_index], u_hist[best_index]

	def run(self, initial_point_list, topk=4):
		"""Initiate the gradient search algorithm. 

		Parameters
		----------
		initial_point_list : list
			list of initial points to select from
		topk : int, optional
			select and run gradient descent on the `topk` first points of 
			`initial_point_list`

		Returns
		-------
		tuple
			best mitigation point and the utility of the best mitigation point

		Raises
		------
		ValueError
			If `topk` is larger than the length of `initial_point_list`.

		Note
		----
		Uses the :mod:`~multiprocessing` package.

		"""
		print("----------------Gradient Search Starting----------------")
		
		if topk > len(initial_point_list):
			raise ValueError("topk {} > number of initial points {}".format(topk, len(initial_point_list)))

		candidate_points = initial_point_list[:topk]
		mitigations = []
		utilities = np.zeros(topk)
		for cp, count in zip(candidate_points, range(topk)):
			if not isinstance(cp, np.ndarray):
				cp = np.array(cp)
			print("Starting process {} of Gradient Descent".format(count+1))
			m, u  = self.gradient_descent(cp)
			mitigations.append(m)
			utilities[count] = u
		best_index = np.argmax(utilities)
		return mitigations[best_index], utilities[best_index]