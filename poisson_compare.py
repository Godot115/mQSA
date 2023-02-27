# -*- coding: utf-8 -*-
# @Time    : 12/16/22 04:28
# @Author  : godot
# @FileName: qsa_to_common.py
# @Project : MAC
# @Software: PyCharm
import math
# -*- coding: utf-8 -*-
# @Time    : 12/14/22 22:41
# @Author  : godot
# @FileName: qsa_test.py
# @Project : MAC
# @Software: PyCharm
import random
import time

import numpy as np

from models.doe_problem import DoeProblem

from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO
from algorithms import QSA
import mealpy.human_based.QSA as QSA2

from models.poisson_model import PoissonModel


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


import csv

with open("poisson_log.csv", "a+") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["algorithm", "theta", "seed", "best_fitness"] + ["best_history_" + str(i) for i in range(2000)])


def write_log(algorithm, theta, seed, best_fitness, best_history):
    with open("poisson_log.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([algorithm, theta, seed, best_fitness] + best_history)


if __name__ == "__main__":

    paras = {
        "epoch": 20,
        "pop_size": 50,
    }
    models = []
    models.append(QSA.mulQSA(**paras))
    models.append(QSA2.OriginalQSA(**paras))
    models.append(DE.BaseDE(strategy=3, **paras, wf=0.5, cr=0.9))
    models.append(DE.SADE(**paras))
    models.append(PSO.OriginalPSO(**paras))
    models.append(GA.BaseGA(**paras))

    for times in range(50):
        try:
            seed = random.randint(0, 100000)
            np.random.seed(seed)
            random.seed(seed)
            thetas = {
                "t1": [0.52, 0.23, 0.91, -0.14, -0.38, 0.80, -0.71],
                "t2": [0.80, -0.21, -0.13, -0.78, 0.16, 0.62, -0.05]
            }
            for theta in thetas:
                print(theta)
                model = PoissonModel(thetas[theta])
                problem_doe = DoeProblem(model, "D-optimal", grid_size=11, lob=[-1, -1, -1], hib=[1, 1, 1],
                                         minmax="max",
                                         dimension=3,
                                         num_of_sup=10, log_to=None)

                starting_positions = create_starting_positions(problem_doe, paras["pop_size"])
                best_fit = float('-inf')
                for pos in starting_positions:
                    best_fit = max(best_fit, problem_doe.fit_func(pos))

                print(time.time())
                for model in models:
                    best_position, best_fitness = model.solve(problem_doe, starting_positions=starting_positions)
                    best = [best_fit]
                    for epoch_best in model.history.list_global_best:
                        best.append(epoch_best[1][0])
                    write_log(model.get_name(), theta, seed, best_fitness, best)
                    print(best_position)
                    print(times, time.time(), model.get_name(), best_fitness, -math.log(abs(best_fitness)))
        except Exception as e:
            print(e)
            continue
