import numpy as np
import multiprocessing
from ezclimate.tools import _pickle_method, _unpickle_method
try:
    import copyreg
except:
    import copy_reg as copyreg
import types

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
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
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed

    """
    def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound, num_feature, utility,
                 fixed_values=None, fixed_indices=None, print_progress=False):
        self.num_feature = num_feature
        self.pop_amount = pop_amount
        self.num_gen = num_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.u = utility
        self.bound = bound
        self.fixed_values = fixed_values
        self.fixed_indices = fixed_indices
        self.print_progress = print_progress

    def _generate_population(self, size):
        """Return 1D-array of random values in the given bound as the initial population."""
        pop = np.random.random([size, self.num_feature])*self.bound
        if self.fixed_values is not None:
            self.fixed_values = self.fixed_values.flatten()
            for ind in pop:
                ind[self.fixed_indices] = self.fixed_values[self.fixed_indices]
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
        for i in range(k):
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
                    child1[self.fixed_indices] = self.fixed_values
                    child2[self.fixed_indices] = self.fixed_values
    
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
        being selected and every element in this individual has a probability `ind_prob` of being
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
                if self.fixed_indices is not None and j in self.fixed_indices:
                    continue
                else:
                    pop[i][j] = max(0.0, pop[i][j]+(np.random.random()-0.5)*scale)
    
    def _uniform_mutation(self, pop, ind_prob, scale=2.0):
        """Mutates individual's elements. The individual has a probability of `mut_prob` of
        being selected and every element in this individual has a probability `ind_prob` of being
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
                self.fixed_values = self.fixed_values.flatten()
                pop[i][self.fixed_indices] = self.fixed_values[self.fixed_indices]

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
    iterations : int
        maximum number of iterations
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
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
    iterations : int
        maximum number of iterations
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed
    scale_alpha : ndarray, optional
        array to scale the learning rate

    """

    def __init__(self, utility, var_nums, accuracy=1e-06, iterations=100, fixed_values=None,
                 fixed_indices=None, print_progress=False, scale_alpha=None):
        self.u = utility
        self.var_nums = var_nums
        self.accuracy = accuracy
        self.iterations = iterations
        self.fixed_values  = fixed_values
        self.fixed_indices = fixed_indices
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

    def numerical_gradient(self, m, delta=1e-08, fixed_indices=None):
        """Calculate utility gradient numerically.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        delta : float, optional
            change in mitigation 
        fixed_indices : ndarray or list, optional
            indices of gradient that should not be calculated

        Returns
        -------
        ndarray
            gradient

        """
        self.delta = delta
        self.m = m
        if fixed_indices is None:
            fixed_indices = []
        grad = np.zeros(len(m))
        if not isinstance(m, np.ndarray):
            self.m = np.array(m)
        pool = multiprocessing.Pool()
        indices = np.delete(list(range(len(m))), fixed_indices)
        res = pool.map(self._partial_grad, indices)
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
            grad = self.numerical_gradient(x_hist[i], fixed_indices=self.fixed_indices)
            m_t = beta1*m_t + (1-beta1)*grad
            v_t = beta2*v_t + (1-beta2)*np.power(grad, 2) 
            m_hat = m_t / (1-beta1**(i+1))
            v_hat = v_t / (1-beta2**(i+1))
            if i != 0:
                accelerator = self._accelerate_scale(accelerator, prev_grad, grad)
            
            new_x = x_hist[i] + ((eta*m_hat)/(np.square(v_hat)+eps)) * accelerator
            new_x[new_x < 0] = 0.0

            if self.fixed_values is not None:
                self.fixed_values = self.fixed_values.flatten()
                new_x[self.fixed_indices] = self.fixed_values[self.fixed_indices]

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
        for cp, count in zip(candidate_points, list(range(topk))):
            if not isinstance(cp, np.ndarray):
                cp = np.array(cp)
            print("Starting process {} of Gradient Descent".format(count+1))
            m, u  = self.gradient_descent(cp)
            mitigations.append(m)
            utilities[count] = u
        best_index = np.argmax(utilities)
        return mitigations[best_index], utilities[best_index]


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
        from scipy.optimize import fmin
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
        nodes = list(range(self.var_nums))
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
