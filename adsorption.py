# -*- coding: utf-8 -*-
# @Time    : 3/7/23 17:25
# @Author  : godot
# @FileName: adsorption.py
# @Project : mQSA
# @Software: PyCharm


import math
import random
import time

import numpy as np

from models.adsorption_model import AbsorptionModel
from models.doe_problem import DoeProblem
from algorithms import QSA
from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO
import mealpy.human_based.QSA as QSA2

from models.potato_model import PotatoModel


def create_starting_positions(problem, pop_size=None):
    lob = problem.lob
    hib = problem.hib
    num_of_sup = problem.num_of_sup
    dimension = problem.dimension
    pos = [[random.uniform(lob[sup % dimension], hib[sup % dimension]) for sup in range(num_of_sup * dimension)] for
           _
           in range(pop_size)]
    weights = np.array([row / sum(row) for row in np.random.uniform(0.1, 1, (pop_size, problem.num_of_sup))])
    starting_positions = np.append(pos, weights, axis=1)
    return starting_positions


if __name__ == "__main__":

    paras = {
        "epoch": 2000,
        "pop_size": 500,
    }
    algorithms = []
    algorithms.append(QSA.mulQSA(**paras))
    algorithms.append(QSA2.OriginalQSA(**paras))
    algorithms.append(DE.BaseDE(strategy=3, **paras, wf=0.5, cr=0.9))
    algorithms.append(DE.SADE(**paras))
    algorithms.append(PSO.OriginalPSO(**paras))
    algorithms.append(GA.BaseGA(**paras))
    for algorithm in algorithms:
        best_fit = -1
        with open("adsorption_test.txt", "a") as file:
            file.write(algorithm.get_name() + "\t")
        for times in range(100):
            # try:
            seed = random.randint(0, 100000)
            np.random.seed(seed)
            random.seed(seed)
            theta = [1.191, 0.3047, -207.3, -0.05659, 0.000054, 711.9, -0.006995, 0.001068]
            model = AbsorptionModel(theta)
            lob = [1.0, 0.5, 0.02]
            hib = [30.0, 400.0, 0.15]
            problem_doe = DoeProblem(model, "D-optimal", grid_size=11, lob=lob, hib=hib, minmax="max",
                                     dimension=3,
                                     num_of_sup=10, log_to=None)

            starting_positions = create_starting_positions(problem_doe, paras["pop_size"])
            best_position, best_fitness = algorithm.solve(problem_doe, starting_positions=starting_positions)
            if best_fitness > best_fit:
                with open("adsorption_test.txt", "r+") as file:
                    write_position = file.seek(0)
                    while True:
                        line_start = file.tell()
                        line = file.readline()
                        if not line:
                            break
                        if line.startswith(algorithm.get_name()):
                            comma_pos = line.find(",", len(algorithm.name))
                            file.seek(line_start + comma_pos + 1)
                            file.write(algorithm.get_name() + "\t")
                            file.write(str(seed) + '\t')
                            file.write(str(best_fitness) + "\n")
                            file.write(str(best_position))
                            file.write('\n')
                            break
                best_fit = best_fitness
            print(best_position)
            print(times, time.time(), algorithm.get_name(), best_fitness, -math.log(abs(best_fitness)))
        # except Exception as e:
        #     print(e)
        #     continue
