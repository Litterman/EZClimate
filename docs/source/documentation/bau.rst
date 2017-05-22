===
BAU
===

The :class:`ezclimate.bau.DLWBusinessAsUsual` provides an analysis framework of business as usual scenario. We assume constant consumption growth and GHG emissions that grow linearly over time without mitigation. For analysis, emission level are given at certain decision time points. Emissions between those decision time points are calcualted using linear interploation. GHG levels are calculated in accordance with the emission path.

Users can create their own business as usual assumptions (e.g., non-linear growth of GHG emission), by writing their own class, inheriting the base class :class:`ezclimate.bau.BusinessAsUsual`. 



