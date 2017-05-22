hard_dependencies = ("numpy", )
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))


__all__ = ["bau", "cost", "damage", "damage_simulation", "utility"
		   "optimization", "forcing", "tools", "tree", "storage_tree", "analysis"]

import optimization
import analysis
