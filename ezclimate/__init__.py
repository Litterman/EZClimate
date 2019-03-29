hard_dependencies = ("numpy", )
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))

from ezclimate.optimization import GeneticAlgorithm, GradientSearch, CoordinateDescent
from ezclimate.analysis import *
from ezclimate.bau import *
from ezclimate.cost import *
from ezclimate.damage import *
from ezclimate.damage_simulation import *
from ezclimate.forcing import *
from ezclimate.storage_tree import *
from ezclimate.tree import *
from ezclimate.utility import *

