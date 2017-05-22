from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod
from storage_tree import BigStorageTree

class Cost(object):
	"""Abstract Cost class for the EZ-Climate model."""
	__metaclass__ = ABCMeta

	@abstractmethod
	def cost(self):
		pass

	@abstractmethod
	def price(self):
		pass


class DLWCost(Cost):
	"""Class to evaluate the cost curve for the EZ-Climate model.

	Parameters
	----------
	tree : `TreeModel` object
		tree structure used
	emit_at_0 : float
		initial GHG emission level
	g : float
		intital scale of the cost function
	a : float
		curvature of the cost function
	join_price : float
		price at which the cost curve is extended
	max_price : float
		price at which carbon dioxide can be removed from atmosphere in unlimited scale
	tech_const : float 
		determines the degree of exogenous technological improvement over time. A number 
			of 1.0 implies 1 percent per yer lower cost
	tech_scale : float
		determines the sensitivity of technological change to previous mitigation
	cons_at_0 : float 
		intital consumption. Default $30460bn based on US 2010 values.

	Attributes
	----------
	tree : `TreeModel` object
		tree structure used
	g : float
		intital scale of the cost function
	a : float
		curvature of the cost function
	max_price : float
		price at which carbon dioxide can be removed from atmosphere in unlimited scale
	tech_const : float 
		determines the degree of exogenous technological improvement over time. A number 
			of 1.0 implies 1 percent per yer lower cost
	tech_scale : float
		determines the sensitivity of technological change to previous mitigation
	cons_at_0 : float 
		intital consumption. Default $30460bn based on US 2010 values.
	cbs_level : float
		constant 
	cbs_deriv : float
		constant
	cbs_b : float
		constant 
	cbs_k : float
		constant
	cons_per_ton : float
		constant 
		
	"""

	def __init__(self, tree, emit_at_0, g, a, join_price, max_price,
				tech_const, tech_scale, cons_at_0):
		self.tree = tree
		self.g = g
		self.a = a
		self.max_price = max_price
		self.tech_const = tech_const
		self.tech_scale = tech_scale
		self.cbs_level = (join_price / (g * a))**(1.0 / (a - 1.0))
		self.cbs_deriv = self.cbs_level / (join_price * (a - 1.0))
		self.cbs_b = self.cbs_deriv * (max_price - join_price) / self.cbs_level
		self.cbs_k = self.cbs_level * (max_price - join_price)**self.cbs_b
		self.cons_per_ton = cons_at_0 / emit_at_0

	def cost(self, period, mitigation, ave_mitigation):
		"""Calculates the mitigation cost for the period. For details about the cost function
		see DLW-paper.

		Parameters
		----------
		period : int 
			period in tree for which mitigation cost is calculated
		mitigation : ndarray
			current mitigation values for period
		ave_mitigation : ndarray
			average mitigation up to this period for all nodes in the period

		Returns
		-------
		ndarray 
			cost

		"""		
		years = self.tree.decision_times[period]
		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100.0))**years
		cbs = self.g * (mitigation**self.a) 
		bool_arr = (mitigation < self.cbs_level).astype(int)
		if np.all(bool_arr):
			c = (cbs * tech_term) / self.cons_per_ton 
		else:
			base_cbs = self.g * self.cbs_level**self.a
			bool_arr2 = (mitigation > self.cbs_level).astype(int)
			extension = ((mitigation-self.cbs_level) * self.max_price 
						- self.cbs_b*mitigation * (self.cbs_k/mitigation)**(1.0/self.cbs_b) / (self.cbs_b-1.0)
						+ self.cbs_b*self.cbs_level * (self.cbs_k/self.cbs_level)**(1.0/self.cbs_b) / (self.cbs_b-1.0))
			
			c = (cbs * bool_arr + (base_cbs + extension)*bool_arr2) * tech_term / self.cons_per_ton
		return c
	
	def price(self, years, mitigation, ave_mitigation):
		"""Inverse of the cost function. Gives emissions price for any given 
		degree of mitigation, average_mitigation, and horizon.

		Parameters
		----------
		years : int y
			years of technological change so far
		mitigation : float 
			mitigation value in node
		ave_mitigation : float
			average mitigation up to this period

		Returns
		-------
		float 
			the price.

		"""
		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100))**years
		if mitigation < self.cbs_level:
			return self.g * self.a * (mitigation**(self.a-1.0)) * tech_term
		else:
			return (self.max_price - (self.cbs_k/mitigation)**(1.0/self.cbs_b)) * tech_term











