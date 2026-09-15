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

import sys        # to allow passing command-line parameters
import random
import csv
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

pattern = []

with open('smiley.txt', newline='\n') as bigall3_file:
    pattern_data = csv.reader(bigall3_file, delimiter=' ')
    for row in pattern_data:
        pattern.append([int(numeric_string) for numeric_string in row])

row_num = pattern_data.line_num

rows = int(np.array(pattern[0]))
cols = int(np.array(pattern[1]))
dim_pattern = rows * cols 

pattern = np.ndarray((1,rows*cols),dtype=int,buffer=np.array(pattern[2:]))
#print(pattern)
#print("\n")
pattern = pattern[0]
#print(pattern)
#print("\n")


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=dim_pattern)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_pat(individual):
    global pattern
    score=0
    for i in range (0,individual.size):
        if individual[i]==pattern[i]:
            score+=1;
    return(score),


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = np.array((1,2,3,4))
        >>> b = np.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    

toolbox.register("evaluate", eval_pat)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main(seed=0):
    random.seed(seed)
    pop = toolbox.population(n=300)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # np.array_equal or np.allclose solve this issue.
    hof = tools.HallOfFame(10, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000, stats=stats, halloffame=hof, verbose=True)

    print("-- End of (successful) evolution --")

    best_ind = hof[0]

    print("\nPattern to be matched                 Best individual")
    for i in range(0,rows) :
        print(pattern[cols*i:cols*i+cols],"     ",best_ind[cols*i:cols*i+cols])

#    print("\nBest individual\n")        
#    for i in range(0,rows) :
#        print(best_ind[cols*i:cols*i+cols-1])
    
    print("\nBest fitness is %s \n" % (best_ind.fitness.values[0]))
    
    return pop, stats, hof

if __name__ == "__main__":
    if len(sys.argv)>1:
        main(sys.argv[1])     # sets the random seed as the
                              # first command-line parameter
    else:
        main()
