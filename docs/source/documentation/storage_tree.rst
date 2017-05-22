============
Storage Tree
============

Values generated during utility calculation, such as damage, cost etc., are stored in the two derived classes of the abstract class :class:`ezclimate.storage_tree.BaseStorageTree` namely, :class:`ezclimate.storage_tree.SmallStorageTree`, and :class:`ezclimate.storage_tree.BigStorageTree`. The 'small' storage tree stores values for every time period where decisions about mitigation are made, and the 'big' storage tree stores values for every subinterval period too. The base class defines the method for initializing the dictionary :attr:`tree` were the values are stored. The keys of the dictionary are the time periods in the tree where values are stored. It also defines methods for getting and setting, saving, and information about periods. Moreover, it defines an abstract method :func:`get_next_period_array` that needs to be initialized in derived classes. 

Small storage tree
------------------
In the :class:`ezclimate.storage_tree.SmallStorageTree` there's no storage in nodes between periods in :attr:`decision_times` - that needs to be defined when initilizing and object of the class. For example, 

.. literalinclude:: ../code/storage_tree.py
   :lines: 1-3

Hence the :obj:`sst` will have 7 keys in its :attr:`tree` dictionary. To access elements in the :attr:`tree` dictionary, the following is equivalent:

.. literalinclude:: ../code/storage_tree.py
   :lines: 5-6


Big storage tree
----------------
In the :class:`ezclimate.storage_tree.BigStorageTree` there's storage in nodes between periods in :attr:`decision_times`. Besides defining the :attr:`decision_times` when initilizing an object of the class, the user also needs to define the length of the subinterval. 

.. literalinclude:: ../code/storage_tree.py
   :lines: 8-10