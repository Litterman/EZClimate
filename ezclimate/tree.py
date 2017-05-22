from __future__ import division
import numpy as np

class TreeModel(object):
    """Tree model for the EZ-Climate model. It provides the structure of a non-recombining tree.

    Parameters
    ----------
    decision_times : ndarray or list
        years in the future where decisions will be made
    prob_scale : float, optional
        scaling constant for probabilities

    Attributes
    ----------
    decision_times : ndarray 
        years in the future where decisions will be made
    prob_scale : float
        scaling constant for probabilities
    node_prob : ndarray
        probability of reaching node from period 0
    final_states_prob : ndarray
        last periods `node_prob`

    """

    def __init__(self, decision_times, prob_scale=1.0):
        self.decision_times = decision_times
        if isinstance(self.decision_times, list):
            self.decision_times = np.array(self.decision_times)
        self.prob_scale = prob_scale
        self.node_prob = None
        self.final_states_prob = None
        self._create_probs()

    @property
    def num_periods(self):
        """int: the number of periods in the tree"""
        return len(self.decision_times)-1

    @property
    def num_decision_nodes(self):
        """int: the number of nodes in tree"""
        return (2**self.num_periods) - 1

    @property
    def num_final_states(self):
        """int: the number of nodes in the last period"""
        return 2**(self.num_periods-1)

    def _create_probs(self):
        """Creates the probabilities of every nodes in the tree structure."""
        self.final_states_prob = np.zeros(self.num_final_states)
        self.node_prob = np.zeros(self.num_decision_nodes)
        self.final_states_prob[0] = 1.0
        sum_probs = 1.0
        next_prob = 1.0

        for n in range(1, self.num_final_states):
            next_prob = next_prob * self.prob_scale**(1.0 / n)
            self.final_states_prob[n] = next_prob
        self.final_states_prob /= np.sum(self.final_states_prob)

        self.node_prob[self.num_final_states-1:] = self.final_states_prob
        for period in range(self.num_periods-2, -1, -1): 
            for state in range(0, 2**period):
                pos = self.get_node(period, state)
                self.node_prob[pos] = self.node_prob[2*pos + 1] + self.node_prob[2*pos + 2]

    def get_num_nodes_period(self, period):
        """Returns the number of nodes in the period.

        Parameters
        ----------
        period : int 
            period

        Returns
        -------
        int 
            number of nodes in period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_num_nodes_period(2)
        4
        >>> t.get_num_nodes_period(5)
        32

        """
        if period >= self.num_periods:
            return 2**(self.num_periods-1)
        return 2**period
    
    def get_nodes_in_period(self, period):
        """Returns the first and last nodes in the period.

        Parameters
        ----------
        period : int 
            period

        Returns
        -------
        int 
            number of nodes in period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_nodes_in_period(0)
        (0, 0)
        >>> t.get_nodes_in_period(1)
        (1, 2)
        >>> t.get_nodes_in_period(4)
        (15, 30)

        """
        if period >= self.num_periods:
            period = self.num_periods-1
        nodes = self.get_num_nodes_period(period)
        first_node = self.get_node(period, 0)
        return (first_node, first_node+nodes-1)

    def get_node(self, period, state):
        """Returns the node in period and state provided.

        Parameters
        ----------
        period : int 
            period
        state : int
            state of the node

        Returns
        -------
        int 
            node number

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_node(1, 1)
        2
        >>> t.get_node(4, 10)
        25
        >>> t.get_node(4, 20)
        ValueError: No such state in period 4

        Raises
        ------
        ValueError
            If period is too large or if the state is too large
            for the period.

        """
        if period > self.num_periods:
            raise ValueError("Given period is larger than number of periods")
        if state >= 2**period:
            raise ValueError("No such state in period {}".format(period))
        return 2**period + state - 1

    def get_state(self, node, period=None):
        """Returns the state the node represents.

        Parameters
        ----------
        node : int
            the node
        period : int, optional
            the period

        Returns
        -------
        int 
            state

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_state(0)
        0
        >>> t.get_state(4, 2)
        1

        """
        if node >= self.num_decision_nodes:
            return node - self.num_decision_nodes
        if not period:
            period = self.get_period(node)
        return node - (2**period - 1)

    def get_period(self, node):
        """Returns what period the node is in.

        Parameters
        ----------
        node : int
            the node

        Returns
        -------
        int 
            period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_period(0)
        0
        >>> t.get_period(4)
        2

        """
        if node >= self.num_decision_nodes: 
            return self.num_periods

        for i in range(0, self.num_periods):
            if int((node+1) / 2**i ) == 1:
                return i

    def get_parent_node(self, child):
        """Returns the previous or parent node of the given child node.

        Parameters
        ----------
        child : int
            the child node

        Returns
        -------
        int 
            partent node

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_parent_node(2)
        0
        >>> t.get_parent_node(4)
        1
        >>> t.get_parent_node(10)
        4

        """
        if child == 0:
            return 0
        if child > self.num_decision_nodes:
            return child - self.num_final_states
        if child % 2 == 0:
            return int((child - 2) / 2)
        else:
            return int((child - 1 ) / 2)

    def get_path(self, node, period=None):
        """Returns the unique path taken to come to given node.

        Parameters
        ----------
        node : int
            the node

        Returns
        -------
        ndarray
            path to get to `node`            

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_path(2)
        array([0, 2])
        >>> t.get_parent_node(4)
        array([0, 1, 4])
        >>> t.get_parent_node(62)
        array([ 0,  2,  6, 14, 30, 62])
        
        """
        if period is None:
            period = self.get_period(node)
        path = [node]
        for i in range(0, period):
            parent = self.get_parent_node(path[i])
            path.append(parent)
        path.reverse()
        return np.array(path)

    def get_probs_in_period(self, period):
        """Returns the probabilities to get from period 0 to nodes in period.

        Parameters
        ----------
        period : int
            the period

        Returns
        -------
        ndarray
            probabilities       

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_probs_in_period(2)
        array([ 0.25,  0.25,  0.25,  0.25])
        >>> t.get_probs_in_period(4)
        array([ 0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
                0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
                0.0625,  0.0625])

        """
        first, last = self.get_nodes_in_period(period)
        return self.node_prob[range(first, last+1)]
    
    def reachable_end_states(self, node, period=None, state=None):
        """Returns what future end states can be reached from given node.

        Parameters
        ----------
        node : int
            the node
        period : int, optional
            the period
        state : int, optional
            the state the node is in

        Returns
        -------
        tuple
            (worst end state, best end state)

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.reachable_end_states(0)
        (0, 31)
        >>> t.reachable_end_states(10)
        (12, 15)
        >>> t.reachable_end_states(32)
        (1, 1)

        """
        if period is None:
            period = self.get_period(node)
        if period >= self.num_periods:
            return (node - self.num_decision_nodes, node - self.num_decision_nodes)
        if state is None:
            state = self.get_state(node, period)

        k = int(self.num_final_states / 2**period)
        return (k*state, k*(state+1)-1)

  

