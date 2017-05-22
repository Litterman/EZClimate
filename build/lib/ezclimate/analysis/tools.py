from __future__ import division, print_function
import numpy as np
from scipy.optimize import brentq
from ezclimate.storage_tree import BigStorageTree, SmallStorageTree
from ezclimate.optimization import GeneticAlgorithm, GradientSearch


def additional_ghg_emission(m, utility):
	"""Calculate the emission added by every node.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	
	Returns
	-------
	ndarray
		additional emission in nodes
	
	"""
	additional_emission = np.zeros(len(m))
	cache = set()
	for node in range(utility.tree.num_final_states, len(m)):
		path = utility.tree.get_path(node)
		for i in range(len(path)):
			if path[i] not in cache:
				additional_emission[path[i]] = (1.0 - m[path[i]]) *  utility.damage.bau.emission_to_ghg[i]
				cache.add(path[i])
	return additional_emission

def store_trees(prefix=None, start_year=2015, **kwargs):
	"""Saves values of `BaseStorageTree` objects. The file is saved into the 'data' directory
	in the current working directory. If there is no 'data' directory, one is created. 

	Parameters
	----------
	prefix : str, optional 
		prefix to be added to file_name
	start_year : int, optional
		start year of analysis
	**kwargs 
		arbitrary keyword arguments of `BaseStorageTree` objects

	"""
	if prefix is None:
		prefix = ""
	for name, tree in kwargs.items():
		tree.write_columns(prefix + "trees", name, start_year)

def delta_consumption(m, utility, cons_tree, cost_tree, delta_m):
	"""Calculate the changes in consumption and the mitigation cost component 
	of consumption when increaseing period 0 mitigiation with `delta_m`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	cons_tree : `BigStorageTree` object
		consumption storage tree of consumption values
		from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost storage tree of cost values from optimal mitigation values
	delta_m : float 
		value to increase period 0 mitigation by
	
	Returns
	-------
	tuple
		(storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	m_copy = m.copy()
	m_copy[0] += delta_m

	new_utility_tree, new_cons_tree, new_cost_tree, new_ce_tree = utility.utility(m_copy, return_trees=True)

	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m
	
	return new_cons_tree, cost_array, new_utility_tree[0]

def constraint_first_period(utility, first_node, m_size):
	"""Calculate the changes in consumption, the mitigation cost component of consumption,
	and new mitigation values when constraining the first period mitigation to `first_node`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	first_node : float
		value to constrain first period to
	
	Returns
	-------
	tuple
		(new mitigation array, storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	fixed_values = np.array([first_node])
	fixed_indicies = np.array([0])
	ga_model = GeneticAlgorithm(pop_amount=150, num_generations=100, cx_prob=0.8, mut_prob=0.5, bound=1.5,
								num_feature=m_size, utility=utility, fixed_values=fixed_values, 
								fixed_indicies=fixed_indicies, print_progress=True)

	gs_model = GradientSearch(var_nums=m_size, utility=utility, accuracy=1e-7,
							  iterations=250, fixed_values=fixed_values, fixed_indicies=fixed_indicies, 
							  print_progress=True)

	final_pop, fitness = ga_model.run()
	sort_pop = final_pop[np.argsort(fitness)][::-1]
	new_m, new_utility = gs_model.run(initial_point_list=sort_pop, topk=1)

	return new_m

def numerical_scc(m, utility, delta_m):
	utility_t, cons_t, cost_t, ce_t = utility.utility(m, return_trees=True)
	m_copy = m.copy()
	m_copy[0] += delta_m
	delta_utility_t, delta_cons_t, delta_cost_t, delta_ce_t = utility.utility(m_copy, return_trees=True)
	delta_utility = (delta_utility_t[0]-utility_t[0])
	node_eps = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
	scc = 0
	for period in cons_t.decision_times[1:]:
		cons_t.tree[period] = (delta_cons_t[period]-cons_t[period])
		for node in range(len(cons_t[period])):
			node_eps.tree[period][node] = delta_m
			adj_utiity = utility.adjusted_utility(m, node_cons_eps=node_eps)
			node_eps.tree[period][node] = 0.0
			cons_t.tree[period][node] = (cons_t[period][node]/(delta_m)) * ((adj_utiity-utility_t[0])/cons_t[period][node])
			cons_t.tree[period][node] = np.nan_to_num(cons_t[period][node])
		scc += cons_t.tree[period].sum()
	scc = scc*((delta_cons_t[0]-cons_t[0])/delta_utility)*delta_m*m[0]
	return scc


def find_ir(m, utility, payment, a=0.0, b=1.0): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""

	def min_func(price):
		utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return utility_with_final_payment - utility_with_initial_payment

	return brentq(min_func, a, b)

def find_term_structure(m, utility, payment, a=0.0, b=1.5): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""

	def min_func(price):
		period_cons_eps = np.zeros(int(utility.decision_times[-1]/utility.period_len) + 1)
		period_cons_eps[-2] = payment
		utility_with_payment = utility.adjusted_utility(m, period_cons_eps=period_cons_eps)
		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return  utility_with_payment - utility_with_initial_payment

	return brentq(min_func, a, b)

def find_bec(m, utility, constraint_cost, a=-0.1, b=1.5):
	"""Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	constraint_cost : float
		utility cost of constraining period 0 to zero
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""

	def min_func(delta_con):
		base_utility = utility.utility(m)
		new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
		print(base_utility, new_utility, constraint_cost)
		return new_utility - base_utility - constraint_cost

	return brentq(min_func, a, b)

def perpetuity_yield(price, start_date, a=0.1, b=10.0):
	"""Find the yield of a perpetuity starting at year `start_date`.

	Parameters
	----------
	price : float
		price of bond ending at `start_date`
	start_date : int
		start year of perpetuity
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	Note
	----
	requires the 'scipy' package

	"""
	
	def min_func(perp_yield):
		return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

	try:
		return brentq(min_func, a, b)
	except:
		return 1e-11