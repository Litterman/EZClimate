from __future__ import division
import numpy as np


class Forcing(object):
	"""Radiative forcing for the EZ-Climate model. Determines the excess energy created 
	by GHGs in the atmosphere. 

	Attributes
	----------
	sink_start : float
		sinking constant
	forcing_start : float
		forcing start constant
	forcing_p1 : float
		forcing constant
	forcing_p2 : float
		forcing constant 
	forcing_p3 : float
		forcing constant
	absorbtion_p1 : float
		absorbtion constant
	absorbtion_p2 : float 
		absorbtion constant
	lsc_p1 : float
		class constant
	lsc_p2 : float
		class constant 

	"""
	sink_start = 35.596
	forcing_start = 4.926
	forcing_p1 = 0.13173
	forcing_p2 = 0.607773
	forcing_p3 = 315.3785
	absorbtion_p1 = 0.94835
	absorbtion_p2 = 0.741547
	lsc_p1 = 285.6268
	lsc_p2 = 0.88414

	@classmethod
	def forcing_and_ghg_at_node(cls, m, node, tree, bau, subinterval_len, returning="forcing"):
		"""Calculates the radiative forcing based on GHG evolution leading up to the 
		damage calculation in `node`.

		Parameters
		----------
		m : ndarray
			array of mitigations
		node : int
			node for which forcing is to be calculated
		tree : `TreeModel` object 
			tree structure used
		bau : `BusinessAsUsual` object
			business-as-usual scenario of emissions
		subinterval_len : float
			subinterval length
		returning : string, optional
			* "forcing": implies only the forcing is returned
			* "ghg": implies only the GHG level is returned
			* "both": implies both the forcing and GHG level is returned

        Returns
        -------
        tuple or float
        	if `returning` is 
        		* "forcing": only the forcing is returned
        		* "ghg": only the GHG level is returned
        		* "both": both the forcing and GHG level is returned 

		"""
		if node == 0:
			if returning == "forcing":
				return 0.0
			elif returning== "ghg":
				return bau.ghg_start
			else:
				return 0.0, bau.ghg_start

		period = tree.get_period(node)
		path = tree.get_path(node, period)

		period_lengths = tree.decision_times[1:period+1] - tree.decision_times[:period]
		increments = period_lengths/subinterval_len

		cum_sink = cls.sink_start
		cum_forcing = cls.forcing_start
		ghg_level = bau.ghg_start

		for p in range(0, period):
			start_emission = (1.0 - m[path[p]]) * bau.emission_by_decisions[p]
			if p < tree.num_periods-1: 
				end_emission = (1.0 - m[path[p]]) * bau.emission_by_decisions[p+1]
			else:
				end_emission = start_emission
			increment = int(increments[p])
			for i in range(0, increment):
				p_co2_emission = start_emission + i * (end_emission-start_emission) / increment
				p_co2 = 0.71 * p_co2_emission 
				p_c = p_co2 / 3.67 
				add_p_ppm = subinterval_len * p_c / 2.13
				lsc = cls.lsc_p1 + cls.lsc_p2 * cum_sink
				absorbtion = 0.5 * cls.absorbtion_p1 * np.sign(ghg_level-lsc) * np.abs(ghg_level-lsc)**cls.absorbtion_p2
				cum_sink += absorbtion
				cum_forcing += cls.forcing_p1*np.sign(ghg_level-cls.forcing_p3)*np.abs(ghg_level-cls.forcing_p3)**cls.forcing_p2
				ghg_level += add_p_ppm - absorbtion

		if returning == "forcing":
			return cum_forcing
		elif returning == "ghg":
			return ghg_level
		else:
			return cum_forcing, ghg_level
	
	@classmethod
	def forcing_at_node(cls, m, node, tree, bau, subinterval_len):
		"""Calculates the forcing based mitigation leading up to the 
		damage calculation in `node`.

		Parameters
		----------
		m : ndarray 
			array of mitigations in each node. 
		node : int 
			the node for which the forcing is being calculated.

		Returns
		-------
		float 
			forcing 

		"""

		return cls.forcing_and_ghg_at_node(m, node, tree, bau, subinterval_len, returning="forcing")

	@classmethod
	def ghg_level_at_node(cls, m, node, tree, bau, subinterval_len):
		"""Calculates the GHG level leading up to the damage calculation in `node`.

		Parameters
		----------
		m : ndarray 
			array of mitigations in each node. 
		node : int 
			the node for which the GHG level is being calculated.

		Returns
		-------
		float 
			GHG level at node

		"""
		return cls.forcing_and_ghg_at_node(m, node,tree, bau, subinterval_len, returning="ghg")
		
