=======
Utility
=======

In :class:`ezclimate.utility.EZUtility`, we calculate the utility under the Epstein-Zin framwork. Functions of calculating mariginal utiltiy are also provided for analysis purpose. 

An agent maximizes lifetime utility at each time and for each state of nature by choosing the optimal path of mitigation dependent on Earth’s fragility. Since all uncertainty has been resolved at the final period, we calculate the utility from back forward.

-------------------
Utility Calculation
-------------------
Firstly, we calculate the the utility in the final period, which, in our base case, is the period starting in 2400, the agent receives the utility from all consumption from time T forward. The resulting final-period utility is:

.. math::

   U_T = \left[\frac{1-\beta}{1-\beta(1+r)^{\rho}}\right]^{\frac{1}{\rho}}C_T

In this specification, :math:`(1-\beta)/ \beta` is the pure rate of time preference. The parameter :math:`\rho` measures the agent’s willingness to substitute consumption across time. :math:`C_T` measures the consumption at time T.

Then we calculate the utilties of all periods from back forward given mitigation path.

.. math::

   U_t = \left[(1-\beta){c_t}^{\rho} + \beta \left[\mu_t(\tilde U_{t+1}) \right]^{\rho} \right]^{\frac{1}{\rho}}


where :math:`\mu_t(\tilde U_{t+1})` is the certainty-equivalent of future lifetime utility, based on the agent’s information at time t, and is given by:

.. math::

   \mu_t(\tilde U_{t+1}) = \left( E_t\left[{U_{t+1}}^{\alpha} \right] \right)^{\frac{1}{\alpha}}

:math:`\alpha` captures the agent’s willingness to substitute consumption across (uncertain) future consumption streams. The higher :math:`\alpha` is, the more willing the agent is to substitute consumption across states of nature at a given point in time. 

-------------
Penalty
-------------

When the GHG levels are below 280, penalties cost are imposed. The penalties in previous nodes in the path leading to the current node is summed and added to current period's penalty, given by

.. math::

   \max \left( 0, \min \left( \frac{280-GHG\ level}{GHG\ level}, max\ penalty \right) \right)

