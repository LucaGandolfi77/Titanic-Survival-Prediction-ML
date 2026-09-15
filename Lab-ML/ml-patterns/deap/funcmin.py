#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes a function of 3 real arguments
#    within the range [0,10]

import random
import numpy
import time

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_real' to be an attribute ('gene')
#                      which corresponds to real numbers sampled uniformly
#                      from the range [0,10]
toolbox.register("attr_real", random.uniform, -250, 250)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 3 'attr_real' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_real, 3)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be minimized

def evalfun(individual):
    val = (1.5 + numpy.sin(individual[2])) * (1+ ((20-individual[0])**2 + (30-individual[1])**2)**0.5)
    return val,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalfun)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# defines the statistics to be collected
stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

#----------

def main():
    random.seed(time.time())

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN is the number of generations
    
    CXPB, MUTPB, NGEN = 0.5, 0.2, 1000
    
    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)

    print("-- End of (successful) evolution --")

    best_ind = hof[0]
    
    print("Best individual is %s " % best_ind)
    print("Best fitness is %s " % best_ind.fitness.values[0])

    return pop, log, hof
    

if __name__ == "__main__":
    pop, log, hof = main()
