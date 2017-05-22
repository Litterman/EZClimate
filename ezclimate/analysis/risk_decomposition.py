from __future__ import division, print_function
import numpy as np
from ezclimate.tools import write_columns_csv, append_to_existing
from ezclimate.storage_tree import BigStorageTree
import ezclimate.analysis.tools

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

		self.delta_cons_tree, self.delta_cost_array, delta_utility = tools.delta_consumption(m, self.utility, cons_tree, cost_tree, 0.01)
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
		end_price = tools.find_term_structure(m, self.utility, 0.01)
		perp_yield = tools.perpetuity_yield(end_price, self.sdf_tree.periods[-2])

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
		
		tools.store_trees(prefix=prefix, SDF=self.sdf_tree, DeltaConsumption=self.delta_cons_tree)

		