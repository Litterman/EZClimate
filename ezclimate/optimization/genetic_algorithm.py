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

class GeneticAlgorithm(object):
	"""Optimization algorithm for the EZ-Climate model. 

	Parameters
	----------
	pop_amount : int
		number of individuals in the population
	num_feature : int 
		number of elements in each individual, i.e. number of nodes in tree-model
	num_generations : int 
		number of generations of the populations to be evaluated
	bound : float
		upper bound of mitigation in each node
	cx_prob : float
		 probability of mating
	mut_prob : float
		probability of mutation.
	utility : `Utility` object
		object of utility class
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed

	Attributes
	----------
	pop_amount : int
		number of individuals in the population
	num_feature : int 
		number of elements in each individual, i.e. number of nodes in tree-model
	num_generations : int 
		number of generations of the populations to be evaluated
	bound : float
		upper bound of mitigation in each node
	cx_prob : float
		 probability of mating
	mut_prob : float
		probability of mutation.
	u : `Utility` object
		object of utility class
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed

	"""
	def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound, num_feature, utility,
				 fixed_values=None, fixed_indicies=None, print_progress=False):
		self.num_feature = num_feature
		self.pop_amount = pop_amount
		self.num_gen = num_generations
		self.cx_prob = cx_prob
		self.mut_prob = mut_prob
		self.u = utility
		self.bound = bound
		self.fixed_values = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress

	def _generate_population(self, size):
		"""Return 1D-array of random values in the given bound as the initial population."""
		pop = np.random.random([size, self.num_feature])*self.bound
		if self.fixed_values is not None:
			for ind in pop:
				ind[self.fixed_indicies] = self.fixed_values
		return pop

	def _evaluate(self, indvidual):
		"""Returns the utility of given individual."""
		return self.u.utility(indvidual)

	def _select(self, pop, rate):
		"""Returns a 1D-array of selected individuals.
	    
	    Parameters
	    ----------
	    pop : ndarray 
	    	population given by 2D-array with shape ('pop_amount', 'num_feature')
	    rate : float 
	    	the probability of an individual being selected
		    
	    Returns
	    -------
	    ndarray 
	    	selected individuals

		"""
		index = np.random.choice(self.pop_amount, int(rate*self.pop_amount), replace=False)
		return pop[index,:]

	def _random_index(self, individuals, size):
		"""Generate a random index of individuals of size 'size'.

		Parameters
		----------
		individuals : ndarray or list
			2D-array of individuals
		size : int 
			number of indices to generate
		
		Returns
		-------
		ndarray 
			1D-array of indices

		"""
		inds_size = len(individuals)
		return np.random.choice(inds_size, size)

	def _selection_tournament(self, pop, k, tournsize, fitness):
	    """Select `k` individuals from the input `individuals` using `k`
	    tournaments of `tournsize` individuals.
	    
	    Parameters
	    ----------
	    individuals : ndarray or list
	    	2D-array of individuals to select from
	    k : int
	    	 number of individuals to select
	    tournsize : int
	    	number of individuals participating in each tournament
	   
	   	Returns
	   	-------
	   	ndarray s
	   		selected individuals
	    
	    """
	    chosen = []
	    for i in xrange(k):
	        index = self._random_index(pop, tournsize)
	        aspirants = pop[index]
	        aspirants_fitness = fitness[index]
	        chosen_index = np.where(aspirants_fitness == np.max(aspirants_fitness))[0]
	        if len(chosen_index) != 0:
	        	chosen_index = chosen_index[0]
	        chosen.append(aspirants[chosen_index])
	    return np.array(chosen)

	def _two_point_cross_over(self, pop):
		"""Performs a two-point cross-over of the population.
	    
	    Parameters
	    ----------
		pop : ndarray 
			population given by 2D-array with shape ('pop_amount', 'num_feature')

		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			if np.random.random() <= self.cx_prob:
				cxpoint1 = np.random.randint(1, self.num_feature)
				cxpoint2 = np.random.randint(1, self.num_feature - 1)
				if cxpoint2 >= cxpoint1:
					cxpoint2 += 1
				else: # Swap the two cx points
					cxpoint1, cxpoint2 = cxpoint2, cxpoint1
				child1[cxpoint1:cxpoint2], child2[cxpoint1:cxpoint2] \
				= child2[cxpoint1:cxpoint2].copy(), child1[cxpoint1:cxpoint2].copy()
				if self.fixed_values is not None:
					child1[self.fixed_indicies] = self.fixed_values
					child2[self.fixed_indicies] = self.fixed_values
	
	def _uniform_cross_over(self, pop, ind_prob):
		"""Performs a uniform cross-over of the population.
	    
	    Parameters
	    ----------
	    pop : ndarray
	    	population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float
	    	probability of feature cross-over
	    
		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			size = min(len(child1), len(child2))
			for i in range(size):
				if np.random.random() < ind_prob:
					child1[i], child2[i] = child2[i], child1[i]

	def _mutate(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability of `mut_prob` of 
		beeing selected and every element in this individual has a probability `ind_prob` of beeing 
		mutated. The mutated value is a random number.

		Parameters
		----------
		pop : ndarray
			population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float 
	    	probability of feature mutation 
	    scale : float 
	    	scaling constant of the random generated number for mutation

		"""
		pop_tmp = np.copy(pop)
		mutate_index = np.random.choice(self.pop_amount, int(self.mut_prob * self.pop_amount), replace=False)
		for i in mutate_index:
			feature_index = np.random.choice(self.num_feature, int(ind_prob * self.num_feature), replace=False)
			for j in feature_index:
				if self.fixed_indicies is not None and j in self.fixed_indicies:
					continue
				else:
					pop[i][j] = max(0.0, pop[i][j]+(np.random.random()-0.5)*scale)
	
	def _uniform_mutation(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability of `mut_prob` of
		beeing selected and every element in this individual has a probability `ind_prob` of beeing 
		mutated. The mutated value is the current value plus a scaled uniform [-0.5,0.5] random value.

		Parameters
		----------
		pop : ndarray
			population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float 
	    	probability of feature mutation 
	    scale : float 
	    	scaling constant of the random generated number for mutation

	    """ 
		pop_len = len(pop)
		mutate_index = np.random.choice(pop_len, int(self.mut_prob * pop_len), replace=False)
		for i in mutate_index:
			prob = np.random.random(self.num_feature) 
			inc = (np.random.random(self.num_feature) - 0.5)*scale
			pop[i] += (prob > (1.0-ind_prob)).astype(int)*inc
			pop[i] = np.maximum(1e-5, pop[i])
			if self.fixed_values is not None:
				pop[i][self.fixed_indicies] = self.fixed_values

	def _show_evolution(self, fits, pop):
		"""Print statistics of the evolution of the population."""
		length = len(pop)
		mean = fits.mean()
		std = fits.std()
		min_val = fits.min()
		max_val = fits.max()
		print (" Min {} \n Max {} \n Avg {}".format(min_val, max_val, mean))
		print (" Std {} \n Population Size {}".format(std, length))
		print (" Best Individual: ", pop[np.argmax(fits)])

	def _survive(self, pop_tmp, fitness_tmp):
		"""The 80 percent of the individuals with best fitness survives to
		the next generation.

		Parameters
		----------
		pop_tmp : ndarray
			population
		fitness_tmp : ndarray
			fitness values of `pop_temp`

		Returns
		-------
		ndarray 
			individuals that survived

		"""
		index_fits  = np.argsort(fitness_tmp)[::-1]
		fitness = fitness_tmp[index_fits]
		pop = pop_tmp[index_fits]
		num_survive = int(0.8*self.pop_amount) 
		survive_pop = np.copy(pop[:num_survive])
		survive_fitness = np.copy(fitness[:num_survive])
		return np.copy(survive_pop), np.copy(survive_fitness)

	def run(self):
		"""Start the evolution process.
		
		The evolution steps are:
			1. Select the individuals to perform cross-over and mutation.
			2. Cross over among the selected candidate.
			3. Mutate result as offspring.
			4. Combine the result of offspring and parent together. And selected the top 
			   80 percent of original population amount.
			5. Random Generate 20 percent of original population amount new individuals 
			   and combine the above new population.

		Returns
		-------
		tuple
			final population and the fitness for the final population

		Note
		----
		Uses the :mod:`~multiprocessing` package.

		"""
		print("----------------Genetic Evolution Starting----------------")
		pop = self._generate_population(self.pop_amount)
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		fitness = pool.map(self._evaluate, pop) # how do we know pop[i] belongs to fitness[i]?
		fitness = np.array([val[0] for val in fitness])
		u_hist = np.zeros(self.num_gen)
		for g in range(0, self.num_gen):
			print ("-- Generation {} --".format(g+1))
			pop_select = self._select(np.copy(pop), rate=1)
			
			self._uniform_cross_over(pop_select, 0.50)
			self._uniform_mutation(pop_select, 0.25, np.exp(-float(g)/self.num_gen)**2)
			#self._mutate(pop_select, 0.05)
			
			fitness_select = pool.map(self._evaluate, pop_select)
			fitness_select = np.array([val[0] for val in fitness_select])
			
			pop_tmp = np.append(pop, pop_select, axis=0)
			fitness_tmp = np.append(fitness, fitness_select, axis=0)

			pop_survive, fitness_survive = self._survive(pop_tmp, fitness_tmp)

			pop_new = self._generate_population(self.pop_amount - len(pop_survive))
			fitness_new = pool.map(self._evaluate, pop_new)
			fitness_new = np.array([val[0] for val in fitness_new])

			pop = np.append(pop_survive, pop_new, axis=0)
			fitness = np.append(fitness_survive, fitness_new, axis=0)
			if self.print_progress:
				self._show_evolution(fitness, pop)
			u_hist[g] = fitness[0]

		fitness = pool.map(self._evaluate, pop)
		fitness = np.array([val[0] for val in fitness])
		return pop, fitness