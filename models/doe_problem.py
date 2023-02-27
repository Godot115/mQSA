# -*- coding: utf-8 -*-
# @Time    : 12/15/22 20:40
# @Author  : godot
# @FileName: doe_problem.py
# @Project : MAC
# @Software: PyCharm
import math
from collections import defaultdict

import numpy as np
from mealpy.swarm_based import PSO
from mealpy.utils.problem import Problem
from models.model_util import ModelUtil


class DoeProblem(Problem):
    def __init__(self, model, criterion, grid_size, lob, hib, minmax, dimension, num_of_sup, **kwargs):
        # super(DoeProblem, self).__init__(model=model, criterion=criterion, lb=lb, hb=ub, grid_size=grid_size,
        #                                  minmax=minmax, **kwargs)
        # lb = lob * num_of_sup * dimension + [0.1, ] * num_of_sup
        # ub = hib * num_of_sup * dimension + [1, ] * num_of_sup,
        lb = lob * num_of_sup + [0.1, ] * num_of_sup
        ub = hib * num_of_sup + [1, ] * num_of_sup,

        # lb = [1, 1, -(1 / 3)] + [0.1, ] * num_of_sup
        # ub = [1, -(1 / 3), 1] + [1, ] * num_of_sup
        self.dimension = dimension
        self.num_of_sup = num_of_sup
        self.lob = lob
        self.hib = hib
        self.model_util = ModelUtil(model, criterion, lob, hib, grid_size)
        self.model = model
        super().__init__(lb=lb, ub=ub, minmax=minmax, **kwargs)

    def fit_func(self, solution):
        dimension = self.dimension
        num_of_sup = self.num_of_sup
        points_len = num_of_sup * dimension

        m = self.model_util.m
        points = []
        point = tuple()
        for p in solution[:points_len]:
            point += (p,)
            if len(point) == dimension:
                points.append(point)
                point = tuple()
        weights = solution[points_len:]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        weights = [float(w) for w in weights]
        self.inf_mat = np.zeros((m, m))
        for i in range(len(points)):
            self.inf_mat += self.model.par_deriv_vec(points[i]) * \
                            self.model.par_deriv_vec(points[i]).T * weights[i]
        return np.linalg.det(self.inf_mat) * (1 - abs(1 - weights_sum))
        # return np.linalg.det(self.inf_mat)

    def mul_process(self, solution):
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        design_points = []
        point = tuple()
        points_len = self.num_of_sup * self.dimension
        max_gradient = float("-inf")
        for p in solution[0][:points_len]:
            point += (p,)
            if len(point) == self.dimension:
                design_points.append(point)
                point = tuple()
        weights = defaultdict(lambda: 0.0)
        for idx in range(len(design_points)):
            weights[design_points[idx]] = solution[0][points_len + idx]
        model_util.cal_inf_mat_w(design_points, weights)
        gradients = [0] * len(design_points)
        denominator = 0
        for idx in range(len(design_points)):
            gradients[idx] = gradient_func(design_points[idx])
            max_gradient = max(gradients[idx], max_gradient)
            denominator += weights[design_points[idx]] * gradients[idx]
        for idx in range(len(design_points)):
            weights[design_points[idx]] *= gradients[idx] / denominator
        for idx in range(len(design_points)):
            solution[0][points_len + idx] = weights[design_points[idx]]
        fit = self.fit_func(solution[0])
        solution[1] = [fit, [fit]]
        return solution

    def mul_process_pso(self, solution):
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        design_points = []
        point = tuple()
        points_len = self.num_of_sup * self.dimension
        for p in solution[:points_len]:
            point += (p,)
            if len(point) == self.dimension:
                design_points.append(point)
                point = tuple()
        weights = defaultdict(lambda: 0.0)
        for idx in range(len(design_points)):
            weights[design_points[idx]] = solution[points_len + idx]
        model_util.cal_inf_mat_w(design_points, weights)
        gradients = [0] * len(design_points)
        denominator = 0
        for idx in range(len(design_points)):
            gradients[idx] = gradient_func(design_points[idx])
            denominator += weights[design_points[idx]] * gradients[idx]
        for idx in range(len(design_points)):
            weights[design_points[idx]] *= gradients[idx] / denominator
        for idx in range(len(design_points)):
            solution[points_len + idx] = weights[design_points[idx]]
        fit = self.fit_func(solution)
        return solution, fit

    def exchange_process(self, solution):
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        design_points = []
        point = tuple()
        points_len = self.num_of_sup * self.dimension
        for p in solution[0][:points_len]:
            point += (p,)
            if len(point) == self.dimension:
                design_points.append(point)
                point = tuple()
        weights = defaultdict(lambda: 0.0)
        for idx in range(len(design_points)):
            weights[design_points[idx]] = solution[0][points_len + idx]
        print(design_points)
        for point in design_points:
            star = model_util.generate_star_set(point)
