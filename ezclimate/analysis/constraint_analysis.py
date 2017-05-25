from __future__ import division, print_function
from ezclimate.tools import write_columns_csv, import_csv
import ezclimate.analysis.tools

class ConstraintAnalysis(object):

	def __init__(self, run_name, utility, cfp_m, opt_m=None):
		self.run_name = run_name
		self.utility = utility 
		self.cfp_m = cfp_m
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
		return data[:, 0] #optimal m

	def _constraint_cost(self):
		opt_u = self.utility.utility(self.opt_m)
		cfp_u = self.utility.utility(self.cfp_m)
		return opt_u - cfp_u

	def _delta_consumption(self):
		return tools.find_bec(self.cfp_m, self.utility, self.con_cost)

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
						   "Marginal Benefit Emissions Reduction", "Marginal Cost Emission Reduction"] )
