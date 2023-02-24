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

# model = Model3Negative([349.0268, 1067.0434, 0.7633, 2.6055])
# restrictions = [[0, 2500]]
#

# k = 3
# model = CustomModel("(t_1*x_1*x_1 + t_2*x_2*x_2 + t_3*x_1*x_2 + t_4*x_1*x_2*e**x_3)/(t_5**x_4)",
#                     ["x_1", "x_2", "x_3", "x_4"],
#                     ["t_1", "t_2", "t_3", "t_4", "t_5"],
#                     [1, 1, 1, 1, 1])
# lb = [0.01, 0.01, 0.01, 0.01]
# hb = [10, 10, 10, 10]
# grid_size = 11
# model_util = ModelUtil(model, "D-optimal", lb, hb, grid_size)
# dimension = 4
# num_of_sup = 5

# model = Model2([349.0268, 1067.0434, 0.7633, 2.6055])
# model_util = ModelUtil(model, "D-optimal", lb=[0.01], hb=[2500], grid_size=1001)
# dimension = 1
# num_of_sup = 4


# model = CustomModel("t_1 *x/(x+t_2)",
#                     ["x"],
#                     ["t_1", "t_2"],
#                     [1, 1])
# model_util = ModelUtil(model, "D-optimal", lb=[0.01], hb=[2000], grid_size=1001)
# dimension = 1
# num_of_sup = 2


# model = CustomModel("t_0+t_1 * e**(x/t_2)",
#                     ["x"],
#                     ["t_0", "t_1", "t_2"],
#                     [1, 1, 200])

# model = CustomModel("t_0+(t_1 *x)/(t_2+x)",
#                     ["x"],
#                     ["t_0", "t_1", "t_2"],
#                     [1, 1, 200])

# model = CustomModel("t_0+t_1 *x_1+t_2*x_2+t_3*x_3 + t_4*x_1 *x_2",
#                     ["x_1", "x_2", "x_3"],
#                     ["t_0", "t_1", "t_2", "t_3", "t_4"],
#                     [1, 1, 1, 1, 1])


model = CustomModel("1/(1+e**(-1*(t_0+t_1 *x_1+t_2*x_2+t_3*x_3 + t_4*x_4 + t_5*x_5 + t_6*x_6 + t_7 * x_7)))",
                    ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"],
                    ["t_0", "t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7"],
                    [1, 1, 1, 1, 1, 1, 1, 1])

# model = CustomModel(
#     "t_0+t_1 *x_1+t_2*x_2+t_3*x_3 + t_4*x_1 *x_2 + t_5*x_1**2* x_3**2 + t_6*x_2**2 *x_3 + t_7*x_1 *x_2 *x_3*x_4",
#     ["x_1", "x_2", "x_3", "x_4"],
#     ["t_0", "t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7"],
#     [1, 1, 1, 1, 1, 1, 1, 1])

# model_util = ModelUtil(model, "D-optimal", lob=[0.01], hib=[2000], grid_size=1001)
problem_doe = DoeProblem(model, "D-optimal", grid_size=1001, lob=[0], hib=[1], minmax="max", dimension=7,
                         num_of_sup=9, log_to=None)

paras = {
    "epoch": 1000,
    "pop_size": 100,
}


def create_starting_positions(problem, pop_size=None):
    pos = np.random.uniform(problem.lob, problem.hib, (pop_size, problem.num_of_sup * problem.dimension))
    weights = np.random.uniform(0.1, 1, (pop_size, problem.num_of_sup))
    starting_positions = np.append(pos, weights, axis=1)
    return starting_positions


if __name__ == "__main__":
    seed = random.randint(0, 100000)
    # seed = 881
    # seed = 401
    # seed = 6392
    # seed = 1404
    # seed = 29302
    # seed = 32133
    np.random.seed(seed)
    random.seed(seed)
    models = []
    models.append(DE.SADE(**paras))
    models.append(GA.BaseGA(**paras))
    models.append(QSA.mQSA(**paras))

    starting_positions = create_starting_positions(problem_doe, paras["pop_size"])
    best_fit = float('-inf')
    for pos in starting_positions:
        best_fit = max(best_fit, problem_doe.fit_func(pos))
    # best_fit is the best fitness of the starting positions

    values = []
    names = []
    bests = []
    print(time.time())
    for model in models:
        best_position, best_fitness = model.solve(problem_doe, starting_positions=starting_positions)
        values.append(best_fitness)
        names.append(model.get_name())
        best = [best_fit]
        for epoch_best in model.history.list_global_best:
            best.append(epoch_best[1][0])
        bests.append([best, model.get_name()])
        print(time.time())

    for value in values:
        print(value)

    epoch_idx = [i for i in range(paras["epoch"] + 1)]
    import matplotlib.pyplot as plt

    linestyles = ['--', '-.', ':', '-']

    for best in bests:
        plt.plot(epoch_idx, best[0], label=best[1])

    plt.title("Global Best Fitness")
    plt.xlabel("epoch")
    plt.ylabel("Function value")
    plt.xticks(epoch_idx)

    plt.legend()
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()

    for name in names:
        print(name)
