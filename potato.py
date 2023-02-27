# -*- coding: utf-8 -*-
# @Time    : 3/7/23 17:25
# @Author  : godot
# @FileName: potato.py
# @Project : mQSA
# @Software: PyCharm


import math
import random
import time

import numpy as np

from models.doe_problem import DoeProblem
from algorithms import QSA

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


import csv

with open("potato.csv", "a+") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["algorithm", "theta", "seed", "best_fitness", "best_history"])


def write_log(algorithm, theta, seed, best_fitness, best_history):
    with open("potato.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([algorithm, theta, seed, best_fitness] + best_history)


if __name__ == "__main__":

    paras = {
        "epoch": 1000,
        "pop_size": 500,
    }


    for times in range(50):
        try:
            seed = random.randint(0, 100000)
            np.random.seed(seed)
            random.seed(seed)
            theta = [0.063, 0.087, 1.604, 0.509, 0.414, 3.349, 0.224, 0.418]
            model = PotatoModel(theta)
            lob = [275.3, 80, 107, 130, 35, 14, 11.7, 3]
            hib = [711.7, 155, 127, 151, 60, 26.2, 19.2, 9.1]
            problem_doe = DoeProblem(model, "D-optimal", grid_size=11, lob=lob, hib=hib, minmax="max",
                                     dimension=8,
                                     num_of_sup=10, log_to="console")

            starting_positions = create_starting_positions(problem_doe, paras["pop_size"])
            best_fit = float('-inf')
            for pos in starting_positions:
                best_fit = max(best_fit, problem_doe.fit_func(pos))
            model = QSA.mulQSA(**paras)
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
