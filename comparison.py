# -*- coding: utf-8 -*-
# @Time    : 12/16/22 04:28
# @Author  : godot
# @FileName: qsa_to_common.py
# @Project : MAC
# @Software: PyCharm


# -*- coding: utf-8 -*-
# @Time    : 12/14/22 22:41
# @Author  : godot
# @FileName: qsa_test.py
# @Project : MAC
# @Software: PyCharm
import random
import time

from mealpy.physics_based import SA
from numpy import array

from models.custom_model import CustomModel
from models.doe_problem import DoeProblem
from models.model2 import Model2
from models.model3_negative import Model3Negative
from models.model3_positive import *
from models.model4 import Model4
from models.model5 import Model5
from models.model_util import ModelUtil

from mealpy.bio_based import SMA
from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO
from mealpy.swarm_based import WOA
from algorithms import QSA


def create_starting_positions(problem, pop_size=None):
    pos = np.random.uniform(problem.lob, problem.hib, (pop_size, problem.num_of_sup * problem.dimension))
    weights = np.random.uniform(0.1, 1, (pop_size, problem.num_of_sup))
    starting_positions = np.append(pos, weights, axis=1)
    return starting_positions

import csv
with open("log.csv", "a+") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["algorithm", "theta", "best_fitness","best_history"])

def write_log(algorithm, theta, best_fitness, best_history):
    with open("log.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([algorithm, theta, best_fitness, best_history])
if __name__ == "__main__":



    paras = {
        "epoch": 5000,
        "pop_size": 100,
    }
    models = []
    models.append(DE.SADE(**paras))
    models.append(GA.BaseGA(**paras))
    models.append(QSA.mQSA(**paras))

    for times in range(5):
        seed = random.randint(0, 100000)
        # seed = 881
        # seed = 401
        # seed = 6392
        # seed = 1404
        # seed = 29302
        # seed = 32133
        np.random.seed(seed)
        random.seed(seed)
        thetas = {"t1": [0.432, 0.217, 0.654, 0.879, 0.157, 0.553, 0.366],
                  "t2": [0.956, 0.725, 0.293, 0.522, 0.144, 0.677, 0.934],
                  "t3": [0.018, 0.953, 0.755, 0.409, 0.083, 0.632, 0.289]}
        for theta in thetas:
            print(theta)
            model = CustomModel("1/(1+e**(-1*(t_0+t_1 *x_1+t_2*x_2+t_3*x_3 + t_4*x_4 + t_5*x_5 + t_6*x_6)))",
                                ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6"],
                                ["t_0", "t_1", "t_2", "t_3", "t_4", "t_5", "t_6"],
                                thetas[theta])

            problem_doe = DoeProblem(model, "D-optimal", grid_size=1001, lob=[0], hib=[1], minmax="max", dimension=6,
                                     num_of_sup=8, log_to=None)
            starting_positions = create_starting_positions(problem_doe, paras["pop_size"])
            best_fit = float('-inf')
            for pos in starting_positions:
                best_fit = max(best_fit, problem_doe.fit_func(pos))
            # best_fit is the best fitness of the starting positions

            print(time.time())
            for model in models:
                best_position, best_fitness = model.solve(problem_doe, starting_positions=starting_positions)
                best = [best_fit]
                for epoch_best in model.history.list_global_best:
                    best.append(epoch_best[1][0])
                write_log(model.get_name(), theta, best_fitness, best)
                print(time.time(),model.get_name())

