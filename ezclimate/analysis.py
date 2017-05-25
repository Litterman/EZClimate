from __future__ import division, print_function
import numpy as np
from scipy.optimize import brentq
from storage_tree import BigStorageTree, SmallStorageTree
from optimization import GeneticAlgorithm, GradientSearch
from tools import write_columns_csv, append_to_existing, import_csv


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

	return brentq(min_func, a, b)


class ClimateOutput(object):
	"""Calculate and save output from the EZ-Climate model.

	Parameters
	----------
	utility : `Utility` object
		object of utility class

	Attributes
	----------
	utility : `Utility` object
		object of utility class
	prices : ndarray
		SCC prices
	ave_mitigations : ndarray
		average mitigations
	ave_emissions : ndarray 
		average emissions
	expected_period_price : ndarray
		expected SCC for the period
 	expected_period_mitigation : ndarray
		expected mitigation for the period
	expected_period_emissions : ndarray
		expected emission for the period

	"""

	def __init__(self, utility):
		self.utility = utility
		self.prices = None
		self.ave_mitigations = None
		self.ave_emissions = None
		self.expected_period_price = None
		self.expected_period_mitigation = None
		self.expected_period_emissions = None
		self.ghg_levels = None

	def calculate_output(self, m):
		"""Calculated values based on optimal mitigation. For every **node** the function calculates and saves
			
			* average mitigation
			* average emission
			* GHG level 
			* SCC 

		as attributes. 

		For every **period** the function also calculates and saves
			
			* expected SCC/price
			* expected mitigation 
			* expected emission 
		
		as attributes.

		Parameters
		----------
		m : ndarray or list
			array of mitigation
	
		"""

		bau = self.utility.damage.bau
		tree = self.utility.tree
		periods = tree.num_periods

		self.prices = np.zeros(len(m))
		self.ave_mitigations = np.zeros(len(m))
		self.ave_emissions = np.zeros(len(m))
		self.expected_period_price = np.zeros(periods)
		self.expected_period_mitigation = np.zeros(periods)
		self.expected_period_emissions = np.zeros(periods)
		additional_emissions = additional_ghg_emission(m, self.utility)
		self.ghg_levels = self.utility.damage.ghg_level(m)

		for period in range(0, periods):
			years = tree.decision_times[period]
			period_years = tree.decision_times[period+1] - tree.decision_times[period]
			nodes = tree.get_nodes_in_period(period)
			num_nodes_period = 1 + nodes[1] - nodes[0]
			period_lens = tree.decision_times[:period+1] 
			
			for node in range(nodes[0], nodes[1]+1):
				path = np.array(tree.get_path(node, period))
				new_m = m[path]
				mean_mitigation = np.dot(new_m, period_lens) / years
				price = self.utility.cost.price(years, m[node], mean_mitigation)
				self.prices[node] = price
				self.ave_mitigations[node] = self.utility.damage.average_mitigation_node(m, node, period)
				self.ave_emissions[node] = additional_emissions[node] / (period_years*bau.emission_to_bau)
			
			probs = tree.get_probs_in_period(period)
			self.expected_period_price[period] = np.dot(self.prices[nodes[0]:nodes[1]+1], probs)
			self.expected_period_mitigation[period] = np.dot(self.ave_mitigations[nodes[0]:nodes[1]+1], probs)
			self.expected_period_emissions[period] = np.dot(self.ave_emissions[nodes[0]:nodes[1]+1], probs)

	def save_output(self, m, prefix=None):
		"""Function to save calculated values in `calculate_output` in the file `prefix` + 'node_period_output' 
		in the 'data' directory in the current working directory. 

		The function also saves the values calculated in the utility function in the file
		`prefix` + 'tree' in the 'data' directory in the current working directory. 

		If there is no 'data' directory, one is created. 

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		prefix : str, optional
			prefix to be added to file_name

		"""
		utility_tree, cons_tree, cost_tree, ce_tree = self.utility.utility(m, return_trees=True)
		
		if prefix is not None:
			prefix += "_" 
		else:
			prefix = ""

		write_columns_csv([m, self.prices, self.ave_mitigations, self.ave_emissions, self.ghg_levels], 
		                   prefix+"node_period_output", ["Node", "Mitigation", "Prices", "Average Mitigation",
		                   "Average Emission", "GHG Level"], [range(len(m))])

		append_to_existing([self.expected_period_price, self.expected_period_mitigation, self.expected_period_emissions],
							prefix+"node_period_output", header=["Period", "Expected Price", "Expected Mitigation",
							"Expected Emission"], index=[range(self.utility.tree.num_periods)], start_char='\n')

		store_trees(prefix=prefix, Utility=utility_tree, Consumption=cons_tree, 
			        Cost=cost_tree, CertainEquivalence=ce_tree)


class RiskDecomposition(object):
	"""Calculate and save analysis of output from the EZ-Climate model.

	Parameters
	----------
	utility : `Utility` object
		object of utility class

	Attributes
	----------
	utility : `Utility` object
		object of utility class
	sdf_tree : `BaseStorageTree` object
		SDF for each node
	expected_damages : ndarray
		expected damages in each period
	risk_premium : ndarray
		risk premium in each period
	expected_sdf : ndarray
		expected SDF in each period
	cross_sdf_damages : ndarray
		cross term between the SDF and damages
	discounted_expected_damages : ndarray
		expected discounted damages for each period
	net_discount_damages : ndarray
		net discount damage, i.e. when cost is also accounted for
	cov_term : ndarray 
		covariance between SDF and damages

	"""

	def __init__(self, utility):
		self.utility = utility
		self.sdf_tree = BigStorageTree(utility.period_len, utility.decision_times)
		self.sdf_tree.set_value(0, np.array([1.0]))

		n = len(self.sdf_tree)
		self.expected_damages = np.zeros(n)
		self.risk_premiums = np.zeros(n)
		self.expected_sdf = np.zeros(n)
		self.cross_sdf_damages = np.zeros(n)
		self.discounted_expected_damages = np.zeros(n)
		self.net_discount_damages = np.zeros(n)
		self.cov_term = np.zeros(n)

		self.expected_sdf[0] = 1.0


	def sensitivity_analysis(self, m):
		"""Calculate sensitivity analysis based on the optimal mitigation. For every sub-period, i.e. the 
		periods given by the utility calculations, the function calculates and saves:
			
			* discount prices
			* net expected damages
			* expected damages
			* discounted expected damages
			* risk premium
			* cross SDF & damages
			* covariance between SDF and damages

		as attributes.

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		utility : `Utility` object
			object of utility class
		prefix : str, optional
			prefix to be added to file_name

		"""

		utility_tree, cons_tree, cost_tree, ce_tree = self.utility.utility(m, return_trees=True)
		cost_sum = 0

		self.delta_cons_tree, self.delta_cost_array, delta_utility = delta_consumption(m, self.utility, cons_tree, cost_tree, 0.01)
		mu_0, mu_1, mu_2 = self.utility.marginal_utility(m, utility_tree, cons_tree, cost_tree, ce_tree)
		sub_len = self.sdf_tree.subinterval_len
		i = 1
		for period in self.sdf_tree.periods[1:]:
			node_period = self.sdf_tree.decision_interval(period)
			period_probs = self.utility.tree.get_probs_in_period(node_period)
			expected_damage = np.dot(self.delta_cons_tree[period], period_probs)
			self.expected_damages[i] = expected_damage
			
			if self.sdf_tree.is_information_period(period-self.sdf_tree.subinterval_len):
				total_probs = period_probs[::2] + period_probs[1::2]
				mu_temp = np.zeros(2*len(mu_1[period-sub_len]))
				mu_temp[::2] = mu_1[period-sub_len]
				mu_temp[1::2] = mu_2[period-sub_len]
				sdf = (np.repeat(total_probs, 2) / period_probs) * (mu_temp/np.repeat(mu_0[period-sub_len], 2))
				period_sdf = np.repeat(self.sdf_tree.tree[period-sub_len],2)*sdf 
			else:
				sdf = mu_1[period-sub_len]/mu_0[period-sub_len]
				period_sdf = self.sdf_tree[period-sub_len]*sdf 

			self.expected_sdf[i] = np.dot(period_sdf, period_probs)
			self.cross_sdf_damages[i] = np.dot(period_sdf, self.delta_cons_tree[period]*period_probs)
			self.cov_term[i] = self.cross_sdf_damages[i] - self.expected_sdf[i]*expected_damage

			self.sdf_tree.set_value(period, period_sdf)

			if i < len(self.delta_cost_array):
				self.net_discount_damages[i] = -(expected_damage + self.delta_cost_array[i, 1]) * self.expected_sdf[i] / self.delta_cons_tree[0]
				cost_sum += -self.delta_cost_array[i, 1] * self.expected_sdf[i] / self.delta_cons_tree[0]
			else:
				self.net_discount_damages[i] = -expected_damage * self.expected_sdf[i] / self.delta_cons_tree[0]

			self.risk_premiums[i] = -self.cov_term[i]/self.delta_cons_tree[0]
			self.discounted_expected_damages[i] = -expected_damage * self.expected_sdf[i] / self.delta_cons_tree[0]
			i += 1

	def save_output(self, m, prefix=None):
		"""Save attributes calculated in `sensitivity_analysis` into the file prefix + `sensitivity_output`
		in the `data` directory in the current working directory.

		Furthermore, the perpetuity yield, the discount factor for the last period is calculated, and SCC,
		expected damage and risk premium for the first period is calculated and saved in into the file
		prefix + `tree` in the `data` directory in the current working directory. If there is no `data` directory, 
		one is created.

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		prefix : str, optional
			prefix to be added to file_name

		"""
		end_price = find_term_structure(m, self.utility, 0.01)
		perp_yield = perpetuity_yield(end_price, self.sdf_tree.periods[-2])

		damage_scale = self.utility.cost.price(0, m[0], 0) / (self.net_discount_damages.sum()+self.risk_premiums.sum())
		scaled_discounted_ed = self.net_discount_damages * damage_scale
		scaled_risk_premiums = self.risk_premiums * damage_scale

		if prefix is not None:
			prefix += "_" 
		else:
			prefix = ""

		write_columns_csv([self.expected_sdf, self.net_discount_damages, self.expected_damages, self.risk_premiums, 
			               self.cross_sdf_damages, self.discounted_expected_damages, self.cov_term, 
			               scaled_discounted_ed, scaled_risk_premiums], prefix + "sensitivity_output",
						   ["Year", "Discount Prices", "Net Expected Damages", "Expected Damages", "Risk Premium",
						    "Cross SDF & Damages", "Discounted Expected Damages", "Cov Term", "Scaled Net Expected Damages",
						    "Scaled Risk Premiums"], [self.sdf_tree.periods.astype(int)+2015]) 

		append_to_existing([[end_price], [perp_yield], [scaled_discounted_ed.sum()], [scaled_risk_premiums.sum()], 
			                [self.utility.cost.price(0, m[0], 0)]], prefix+"sensitivity_output",
			                header=["Zero Bound Price", "Perp Yield", "Expected Damages", "Risk Premium", 
							"SCC"], start_char='\n')
		
		store_trees(prefix=prefix, SDF=self.sdf_tree, DeltaConsumption=self.delta_cons_tree)


class ConstraintAnalysis(object):
	def __init__(self, run_name, utility, const_value, opt_m=None):
		self.run_name = run_name
		self.utility = utility 
		self.cfp_m = constraint_first_period(utility, const_value, utility.tree.num_decision_nodes)
		self.opt_m = opt_m
		if self.opt_m is None:
			self.opt_m = self._get_optimal_m()

		self.con_cost = self._constraint_cost()
		self.delta_u = self._first_period_delta_udiff()

		self.delta_c = self._delta_consumption()
		self.delta_c_billions = self.delta_c * self.utility.cost.cons_per_ton \
								* self.utility.damage.bau.emit_level[0]
		self.delta_emission_gton = self.opt_m[0]*self.utility.damage.bau.emit_level[0]
		self.deadweight = self.delta_c*self.utility.cost.cons_per_ton / self.opt_m[0]

		self.delta_u2 = self._first_period_delta_udiff2()
		self.marginal_benefit = (self.delta_u2 / self.delta_u) * self.utility.cost.cons_per_ton
		self.marginal_cost = self.utility.cost.price(0, self.cfp_m[0], 0)

	def _get_optimal_m(self):
		try:
			header, index, data = import_csv(self.run_name+"_node_period_output")
		except:
			print("No such file for the optimal mitigation..")
		return data[:, 0] 

	def _constraint_cost(self):
		opt_u = self.utility.utility(self.opt_m)
		cfp_u = self.utility.utility(self.cfp_m)
		return opt_u - cfp_u

	def _delta_consumption(self):
		return find_bec(self.cfp_m, self.utility, self.con_cost)

	def _first_period_delta_udiff(self):
		u_given_delta_con = self.utility.adjusted_utility(self.cfp_m, first_period_consadj=0.01)
		cfp_u = self.utility.utility(self.cfp_m)
		return u_given_delta_con - cfp_u

	def _first_period_delta_udiff2(self):
		m = self.cfp_m.copy()
		m[0] += 0.01
		u = self.utility.utility(m)
		cfp_u = self.utility.utility(self.cfp_m)
		return u - cfp_u
		
	def save_output(self, prefix=None):
		if prefix is not None:
			prefix += "_" 
		else:
			prefix = ""

		write_columns_csv([self.con_cost, [self.delta_c], [self.delta_c_billions], [self.delta_emission_gton],
						   [self.deadweight], self.delta_u, self.marginal_benefit, [self.marginal_cost]], 
						   prefix + self.run_name + "_constraint_output",
						  ["Constraint Cost", "Delta Consumption", "Delta Consumption $b", 
						   "Delta Emission Gton", "Deadweight Cost", "Marginal Impact Utility",
						   "Marginal Benefit Emissions Reduction", "Marginal Cost Emission Reduction"])