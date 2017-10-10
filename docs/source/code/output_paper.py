from ezclimate.tree import TreeModel

t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385])

from ezclimate.bau import DLWBusinessAsUsual

bau_default_model = DLWBusinessAsUsual()
bau_default_model.bau_emissions_setup(tree=t)

from ezclimate.cost import DLWCost

c = DLWCost(tree=t, emit_at_0=bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=2000.0,
            max_price=2500.0, tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0)

from ezclimate.damage import DLWDamage

df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000], subinterval_len=5)
df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
                     temp_map=1, temp_dist_params=None, maxh=100.0)

from ezclimate.utility import EZUtility

u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=0.9, ra=7.0, time_pref=0.005)


from ezclimate.optimization import GeneticAlgorithm, GradientSearch
import numpy as np

ga_model = GeneticAlgorithm(pop_amount=150, num_generations=75, cx_prob=0.8, mut_prob=0.5, 
                            bound=2.0, num_feature=63, utility=u, print_progress=True)
gs_model = GradientSearch(var_nums=63, utility=u, accuracy=1e-8, 
                          iterations=200, print_progress=True)
final_pop, fitness = ga_model.run()
sort_pop = final_pop[np.argsort(fitness)][::-1]
m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)

print("SCC: ", c.price(0, m_opt[0], 0))


from ezclimate.tree import TreeModel
from ezclimate.bau import DLWBusinessAsUsual
from ezclimate.cost import DLWCost
from ezclimate.damage import DLWDamage
from ezclimate.utility import EZUtility
from ezclimate.optimization import GeneticAlgorithm, GradientSearch
import numpy as np

def base_case():
	t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385])

	bau_default_model = DLWBusinessAsUsual()
	bau_default_model.bau_emissions_setup(tree=t)

	c = DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=2000.0, max_price=2500.0,
					tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0)

	df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000], subinterval_len=5)
	df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
							 temp_map=1, temp_dist_params=None, maxh=100.0)

	u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=0.9, ra=7.0, time_pref=0.005)

	ga_model = GeneticAlgorithm(pop_amount=150, num_generations=75, cx_prob=0.8, mut_prob=0.5, 
	                            bound=1.5, num_feature=63, utility=u, print_progress=True)
	gs_model = GradientSearch(var_nums=63, utility=u, accuracy=1e-8, 
	                          iterations=200, print_progress=True)

	final_pop, fitness = ga_model.run()
	sort_pop = final_pop[np.argsort(fitness)][::-1]
	m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)

	print("SCC: ", c.price(0, m_opt[0], 0))

if __name__ == "__main__":
	base_case()
