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
import operator
import math
import numpy
import time

from deap import base
from deap import creator
from deap import tools

###################  ALGORITHM PARAMETERS ##############################
RSEED = -1          # random seed (if negative, seed = time.time())
GEN = 300           # number of iterations
S_SIZE = 100        # Swarm size (number of particles)
W = 0.7             # Inertia coefficient
PHI1 = 1.4          # local attraction coefficient
PHI2 = 1.4          # global attraction coefficient
                    #
PSIZE = 3           # Search domain dimension (particle size)
                    #
SMIN = -50          # a particle's minimum allowed speed
                    # (max in the negative direction)
SMAX = 50           # a particle's maximum allowed speed
PMIN = -250         # search domain lower limit (same for all particles) 
PMAX = 250          # search domain upper limit (same for all particles)
########################################################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=None,
               smin=None, smax=None, pmin=None, pmax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part

def updateParticle(part, best, w, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    w_all = (w for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))

# update speed
    part.speed = map(operator.mul, w_all, part.speed)
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))

# check if speed is within allowed range    
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif speed > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)

# update particle's position            
    part[:] = list(map(operator.add, part, part.speed))

# check if position is within the search domain
    for j, pos in enumerate(part):
        if pos < part.pmin:
            part[j] = math.copysign(part.pmin, pos)
        elif pos > part.pmax:
            part[j]= math.copysign(part.pmax, pos)

# the goal ('fitness') function to be minimized
def evalfun(individual):
    val = (1.5 + numpy.sin(individual[2])) * (1+ ((20-individual[0])**2 + (30-individual[1])**2)**0.5)
    return val,


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=PSIZE, pmin=PMIN, pmax=PMAX, smin=SMIN, smax=SMAX)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, w=W, phi1=PHI1, phi2=PHI2)


#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalfun)

# defines the statistics to be collected
stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

#----------

def main():

    if RSEED <0:
        random.seed(time.time())
    else:
        random.seed(RSEED)
        

#    GEN = 1000          # number of iterations (moved as global variable)
#    S_SIZE = 300        # Swarm size (number of particles) (moved as global variable)
    # create an initial population of S_SIZE individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=S_SIZE)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
            
        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pop, logbook, best



if __name__ == "__main__":
    pop, log, best = main()
    print("\n-- End of (successful) evolution --")
    
    print("\nBest individual is %s " % best)
    print("Best fitness is %s \n" % best.fitness.values[0])

 
