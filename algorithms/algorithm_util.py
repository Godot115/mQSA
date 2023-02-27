# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:41
# @Author  : godot
# @FileName: algorithm_util.py
# @Project : pyOED
# @Software: PyCharm
import heapq
import itertools
import random
import sys
from random import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from models.model_util import ModelUtil


# model, design_space, plus_minus_sign, *args
class AlgorithmUtil():
    def __init__(self, model_util: ModelUtil, algorithm: str, eff_threshold: float):
        self.model_util = model_util
        self.candidate_set = model_util.candidate_set
        self.m = model_util.m
        self.k = model_util.model.k
        self.criterion_val = 0
        self.eff = 0
        self.algorithm = algorithm
        self.eff_threshold = eff_threshold
        if algorithm == "rex":
            # randomized exchange algorithm
            self.selected_algorithm = self.rex
        elif algorithm == "cocktail":
            # cocktail algorithm
            self.selected_algorithm = self.cocktail_algorithm
        elif algorithm == "fwa":
            # fedorov-wynn algorithm
            self.selected_algorithm = self.fedorov_wynn
        elif algorithm == "mul":
            # multiplicative algorithm
            self.selected_algorithm = self.multiplicative
        elif algorithm == "vex":
            # vertex exchange algorithm
            self.selected_algorithm = self.vertex_exchange
        elif algorithm == "prop":
            self.selected_algorithm = self.proposed
        elif algorithm == "gex":
            self.selected_algorithm = self.gex
        elif algorithm == "prop2":
            self.selected_algorithm = self.proposed2

    def start(self):
        self.selected_algorithm()

    def cocktail_algorithm(self):
        candidate_set = self.candidate_set
        m = self.m
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        weights = defaultdict(lambda: 0)
        threshold = self.eff_threshold
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        max_gradient = float('inf')
        iter = m
        while 1 - (gradient_target() / max_gradient) > threshold:
            iter += 1
            # VEM step begin
            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = sys.maxsize
            min_g_point = sys.maxsize
            for j in range(len(candidate_set)):
                gradient = gradient_func(candidate_set[j])
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = candidate_set[j]
            for j in range(len(design_points)):
                gradient = gradient_func(design_points[j])
                if gradient < min_gradient:
                    min_gradient = gradient
                    min_g_point = design_points[j]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            w_add = weights[max_g_point]
            w_del = weights[min_g_point]
            alpha_s = model_util.cal_ve_alpha_s(max_g_point, min_g_point, w_add, w_del)
            weights[max_g_point] += alpha_s
            weights[min_g_point] -= alpha_s
            if weights[min_g_point] <= 0:
                if max_g_point in design_points:
                    design_points.pop(design_points.index(max_g_point))
                weights.pop(min_g_point)
            if max_g_point not in design_points:
                design_points.append(max_g_point)
            design_add = model_util.cal_observ_mass(max_g_point)
            model_util.fim_add_mass(design_add, alpha_s)
            design_del = model_util.cal_observ_mass(min_g_point)
            model_util.fim_add_mass(design_del, -alpha_s)
            # VEM step end
            # NNE step begin
            idx = 0
            design_points = sorted(design_points, key=lambda x: x, reverse=False)
            while idx < len(design_points) - 1:
                point_1 = design_points[idx]
                point_2 = design_points[idx + 1]
                weight_1 = weights[point_1]
                weight_2 = weights[point_2]
                alpha_s = model_util.cal_ve_alpha_s(point_2, point_1, weight_1, weight_2)
                alpha_s = min(max(alpha_s, -weight_2), weight_1)
                weights[point_1] -= alpha_s
                if weights[point_1] <= 0:
                    if point_1 in design_points:
                        design_points.pop(idx)
                    weights.pop(point_1)
                    idx -= 1
                weights[point_2] += alpha_s
                if weights[point_2] <= 0:
                    if point_2 in design_points:
                        design_points.pop(idx + 1)
                    weights.pop(point_2)
                    idx -= 1
                design_add = model_util.cal_observ_mass(point_2)
                design_del = model_util.cal_observ_mass(point_1)
                model_util.fim_add_mass(design_add, alpha_s)
                model_util.fim_add_mass(design_del, -alpha_s)
                idx += 1
            # NNE step end

            # MA step begin
            denominator = 0
            gradients = [0] * len(design_points)
            for idx in range(len(design_points)):
                gradients[idx] = gradient_func(design_points[idx])
                max_gradient = max(gradients[idx], max_gradient)
                denominator += weights[design_points[idx]] * gradients[idx]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            for idx in range(len(design_points)):
                weights[design_points[idx]] *= gradients[idx] / denominator
            model_util.cal_inf_mat_w(design_points, weights)
            # MA step end
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient
        print(self.criterion_val)

    def rex(self):
        candidate_set = self.candidate_set
        m = self.m
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        weights = defaultdict(lambda: 0)
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        del_threshold = 1e-14
        gamma = min(m, 4 * m)

        max_gradient = float('-inf')
        min_gradient = float('inf')
        max_g_point = sys.maxsize
        min_g_point = sys.maxsize
        gradients = defaultdict(lambda: float('inf'))
        heap = []
        idx = 0
        for point in candidate_set:
            idx += 1
            gradient = gradient_func(point)
            if len(heap) < gamma:
                heapq.heappush(heap, (gradient, point))
            else:
                if gradient > gradients[heap[0][1]]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (gradient, point))
            if gradient > max_gradient:
                max_gradient = gradient
                max_g_point = point
            if gradient < min_gradient and point in design_points:
                min_gradient = gradient
                min_g_point = point
            gradients[point] = gradient
        while 1 - (gradient_target() / max_gradient) > threshold:
            w_add = weights[max_g_point]
            w_del = weights[min_g_point]
            alpha_s = model_util.cal_ve_alpha_s(max_g_point, min_g_point, w_add, w_del)
            weights[max_g_point] += alpha_s
            weights[min_g_point] -= alpha_s
            if weights[min_g_point] <= del_threshold:
                if max_g_point in design_points:
                    design_points.pop(design_points.index(max_g_point))
                weights.pop(min_g_point)
            if max_g_point not in design_points:
                design_points.append(max_g_point)
            design_add = model_util.cal_observ_mass(max_g_point)
            model_util.fim_add_mass(design_add, alpha_s)
            design_del = model_util.cal_observ_mass(min_g_point)
            model_util.fim_add_mass(design_del, -alpha_s)
            L = [point[1] for point in heap]
            shuffle(L)
            shuffle(design_points)

            if weights[min_g_point] < del_threshold:
                for l in L:
                    for k in design_points:
                        if k == l:
                            continue
                        alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                        if weights[k] - alpha < del_threshold or weights[l] + alpha < del_threshold:
                            weights[k] -= alpha
                            weights[l] += alpha
                            model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                            model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)
            else:
                for l in L:
                    for k in design_points:
                        if k == l:
                            continue
                        alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                        weights[k] -= alpha
                        weights[l] += alpha
                        model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                        model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)

            design_points.clear()
            for point in weights.keys():
                if weights[point] > del_threshold:
                    design_points.append(point)
            model_util.cal_inf_mat_w(design_points, weights)

            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = sys.maxsize
            min_g_point = sys.maxsize
            gradients = defaultdict(lambda: float('inf'))
            heap = []
            for point in candidate_set:
                gradient = gradient_func(point)
                if len(heap) < gamma * m:
                    heapq.heappush(heap, (gradient, point))
                else:
                    if gradient > gradients[heap[0][1]]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (gradient, point))
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = point
                if gradient < min_gradient and point in design_points:
                    min_gradient = gradient
                    min_g_point = point
                gradients[point] = gradient
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def fedorov_wynn(self):
        candidate_set = self.candidate_set
        m = self.m
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        weights = defaultdict(lambda: 0)
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        max_gradient = float('inf')
        threshold = self.eff_threshold
        iter = m
        while 1 - (gradient_target() / max_gradient) > threshold:
            iter += 1
            max_g_point = tuple()
            max_gradient = float('-inf')
            for point in candidate_set:
                gradient = gradient_func(point)
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = point
            alpha = ((max_gradient / m) - 1) / (max_gradient - 1)
            for key in weights.keys():
                weights[key] *= (1 - alpha)
            weights[max_g_point] += alpha
            if max_g_point not in design_points:
                design_points.append(max_g_point)
            design_add = model_util.cal_observ_mass(max_g_point)
            model_util.fim_add_mass(design_add, alpha)
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def multiplicative(self):
        candidate_set = self.candidate_set
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        design_points = candidate_set
        weights = defaultdict(lambda: 1 / len(candidate_set))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        max_gradient = float('inf')
        gradients = [0] * len(candidate_set)
        while 1 - (gradient_target() / max_gradient) > threshold:
            max_gradient = float('-inf')
            denominator = 0
            for idx in range(len(design_points)):
                gradients[idx] = gradient_func(design_points[idx])
                max_gradient = max(gradients[idx], max_gradient)
                denominator += weights[design_points[idx]] * gradients[idx]
            if 1 - (gradient_target() / max_gradient) <= threshold:
                break
            for idx in range(len(design_points)):
                weights[design_points[idx]] *= gradients[idx] / denominator

            model_util.cal_inf_mat_w(design_points, weights)
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def vertex_exchange(self):
        candidate_set = self.candidate_set
        m = self.m
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        weights = defaultdict(lambda: 0)
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        max_gradient = float('inf')
        while 1 - (gradient_target() / max_gradient) > threshold:
            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = tuple()
            min_g_point = tuple()
            for point in candidate_set:
                gradient = gradient_func(point)
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = point
                if gradient < min_gradient and point in design_points:
                    min_gradient = gradient
                    min_g_point = point
            if 1 - (gradient_target() / max_gradient) <= threshold:
                break
            alpha = model_util.cal_ve_alpha_s(max_g_point, min_g_point, weights[max_g_point], weights[min_g_point])
            weights[max_g_point] += alpha
            if weights[max_g_point] <= 0:
                design_points.remove(max_g_point)
                weights.pop(max_g_point)
            else:
                if max_g_point not in design_points:
                    design_points.append(max_g_point)
            weights[min_g_point] -= alpha
            if weights[min_g_point] <= 0:
                design_points.remove(min_g_point)
                weights.pop(min_g_point)
            else:
                if min_g_point not in design_points:
                    design_points.append(min_g_point)
            model_util.cal_inf_mat_w(design_points, weights)
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def gex(self):
        model_util = self.model_util
        m = model_util.m
        candidate_set = self.candidate_set
        N_loc = 50
        gradient_func = self.model_util.gradient_func
        generate_star_set = self.model_util.generate_star_set
        eff_rex = 1e-6
        eff_stop = 1 - 1e-9

        def exploration(design_points, candidate_set, N_loc):
            centers = [random.randint(0, len(candidate_set)) for i in range(N_loc)]
            loc_set = set()
            for idx in centers:
                cur_point = candidate_set[idx]
                cur_g = gradient_func(cur_point)
                while True:
                    star_set = generate_star_set(cur_point)
                    star_max_g = float('-inf')
                    star_max_point = tuple()
                    for star_point in star_set:
                        g = gradient_func(star_point)
                        if g > star_max_g:
                            star_max_g = g
                            star_max_point = star_point
                    if star_max_g > cur_g:
                        cur_g = star_max_g
                        cur_point = star_max_point
                    else:
                        break
                if cur_point not in design_points:
                    loc_set.add(cur_point)
            star_set = set()
            for point in design_points:
                star_set = star_set | generate_star_set(point)
                star_set.add(point)
            return loc_set | star_set

        def rex(exp_set, design_points, weights, eff_rex):
            candidate_set = exp_set
            m = self.m
            model_util = self.model_util
            gradient_func = model_util.gradient_func
            gradient_target = model_util.gradient_target
            model_util.cal_inf_mat_w(design_points, weights)
            threshold = eff_rex
            del_threshold = 1e-14
            gamma = min(m, 4 * m)
            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = sys.maxsize
            min_g_point = sys.maxsize
            gradients = defaultdict(lambda: float('inf'))
            heap = []
            for point in candidate_set:
                gradient = gradient_func(point)
                if len(heap) < gamma:
                    heapq.heappush(heap, (gradient, point))
                else:
                    if gradient > gradients[heap[0][1]]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (gradient, point))
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = point
                if gradient < min_gradient and point in design_points:
                    min_gradient = gradient
                    min_g_point = point
                gradients[point] = gradient

            while 1 - (gradient_target() / max_gradient) > threshold:
                w_add = weights[max_g_point]
                w_del = weights[min_g_point]
                alpha_s = model_util.cal_ve_alpha_s(max_g_point, min_g_point, w_add, w_del)
                weights[max_g_point] += alpha_s
                weights[min_g_point] -= alpha_s
                if weights[min_g_point] <= del_threshold:
                    if max_g_point in design_points:
                        design_points.pop(design_points.index(max_g_point))
                    weights.pop(min_g_point)
                if max_g_point not in design_points:
                    design_points.append(max_g_point)
                design_add = model_util.cal_observ_mass(max_g_point)
                model_util.fim_add_mass(design_add, alpha_s)
                design_del = model_util.cal_observ_mass(min_g_point)
                model_util.fim_add_mass(design_del, -alpha_s)
                L = [point[1] for point in heap]
                shuffle(L)
                shuffle(design_points)
                if weights[min_g_point] < del_threshold:
                    for l in L:
                        for k in design_points:
                            if k == l:
                                continue
                            alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                            if weights[k] - alpha < del_threshold or weights[l] + alpha < del_threshold:
                                weights[k] -= alpha
                                weights[l] += alpha
                                model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                                model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)
                else:
                    for l in L:
                        for k in design_points:
                            if k == l:
                                continue
                            alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                            weights[k] -= alpha
                            weights[l] += alpha
                            model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                            model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)

                design_points.clear()
                for point in weights.keys():
                    if weights[point] > del_threshold:
                        design_points.append(point)
                model_util.cal_inf_mat_w(design_points, weights)

                max_gradient = float('-inf')
                min_gradient = float('inf')
                max_g_point = sys.maxsize
                min_g_point = sys.maxsize
                gradients = defaultdict(lambda: float('inf'))
                heap = []
                for point in candidate_set:
                    gradient = gradient_func(point)
                    if len(heap) < gamma * m:
                        heapq.heappush(heap, (gradient, point))
                    else:
                        if gradient > gradients[heap[0][1]]:
                            heapq.heappop(heap)
                            heapq.heappush(heap, (gradient, point))
                    if gradient > max_gradient:
                        max_gradient = gradient
                        max_g_point = point
                    if gradient < min_gradient and point in design_points:
                        min_gradient = gradient
                        min_g_point = point
                    gradients[point] = gradient
            eff = gradient_target() / max_gradient
            return design_points, weights, eff

        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        weights = defaultdict(lambda: 0)
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        c_val = float('inf')
        while True:
            exp_set = exploration(design_points, candidate_set, N_loc)
            design_points, weights, cur_eff = rex(exp_set, design_points, weights, eff_rex)
            cur_c_val = model_util.get_criterion_val()
            if cur_c_val / c_val > eff_stop:
                break
            c_val = cur_c_val
        self.criterion_val = cur_c_val
        self.design_points = [(point, weights[point]) for point in design_points]
        self.eff = cur_eff



    def proposed(self):
        model_util = self.model_util
        candidate_set = self.candidate_set
        m = self.m
        generate_star_set = model_util.generate_star_set
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        weights = defaultdict(lambda: 0)
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        max_gradient = float('inf')
        iter = 0
        while 1 - (gradient_target() / max_gradient) > threshold:
            iter += 1
            add_points = set()
            max_gradient = float('-inf')
            for point in design_points:
                cur_point = point
                cur_g = gradient_func(cur_point)
                while True:
                    star_set = generate_star_set(cur_point)
                    star_max_g = float('-inf')
                    star_max_point = tuple()
                    for star_point in star_set:
                        g = gradient_func(star_point)
                        if g > star_max_g:
                            star_max_g = g
                            star_max_point = star_point
                    if star_max_g > cur_g:
                        cur_g = star_max_g
                        cur_point = star_max_point
                        if star_max_g > max_gradient:
                            max_gradient = star_max_g
                    else:
                        break
                if cur_point not in design_points:
                    add_points.add(cur_point)
            for key in weights.keys():
                weights[key] *= (1 - (1 / (len(design_points) + len(add_points))))
            for point in add_points:
                alpha = 1 / (len(design_points) + len(add_points))
                weights[point] += alpha
                design_points.append(point)
                model_util.fim_add_mass(model_util.cal_observ_mass(point), alpha)

            denominator = 0
            gradients = dict()
            for idx in range(len(design_points)):
                gradients[design_points[idx]] = gradient_func(design_points[idx])
                max_gradient = max(gradients[design_points[idx]], max_gradient)
                denominator += weights[design_points[idx]] * gradients[design_points[idx]]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            for idx in range(len(design_points)):
                weights[design_points[idx]] *= gradients[design_points[idx]] / denominator
            idx = 0
            while idx < len(design_points):
                if weights[design_points[idx]] <= 1e-14:
                    weights.pop(design_points[idx])
                    design_points.pop(idx)
                else:
                    idx += 1
            model_util.cal_inf_mat_w(design_points, weights)
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def generate_neighbor_set(self, point):
        lb = self.model_util.lb
        hb = self.model_util.hb
        grid_size = self.model_util.grid_size
        intervals = [(hb[idx] - lb[idx]) / (grid_size - 1) for idx in range(len(lb))]
        neighbor_ranges = []
        for idx in range(len(point)):
            neighbor_ranges.append(np.linspace(point[idx] - intervals[idx], point[idx] + intervals[idx], 3))
        neighbor_set = set()
        for neighbor in itertools.product(*neighbor_ranges):
            neighbor_set.add(neighbor)
        if point in neighbor_set:
            neighbor_set.remove(point)
        return neighbor_set

    def proposed2(self):
        model_util = self.model_util
        candidate_set = self.candidate_set
        m = self.m
        generate_star_set = model_util.generate_star_set
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (m - 1))] for i in
                         range(m)]
        weights = defaultdict(lambda: 0)
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        max_gradient = float('inf')
        iter = 0
        while 1 - (gradient_target() / max_gradient) > threshold:
            iter += 1
            add_points = set()
            max_gradient = float('-inf')
            for point in design_points:
                cur_point = point
                cur_g = gradient_func(cur_point)
                while True:
                    neighbor_set = generate_star_set(cur_point)
                    star_max_g = float('-inf')
                    star_max_point = tuple()
                    for star_point in neighbor_set:
                        g = gradient_func(star_point)
                        if g > star_max_g:
                            star_max_g = g
                            star_max_point = star_point
                    if star_max_g > cur_g:
                        cur_g = star_max_g
                        cur_point = star_max_point
                        if star_max_g > max_gradient:
                            max_gradient = star_max_g
                    else:
                        break
                if cur_point not in design_points:
                    add_points.add(cur_point)
            for key in weights.keys():
                weights[key] *= (1 - (1 / (len(design_points) + len(add_points))))
            for point in add_points:
                alpha = 1 / (len(design_points) + len(add_points))
                weights[point] += alpha
                design_points.append(point)
                model_util.fim_add_mass(model_util.cal_observ_mass(point), alpha)
            # design_points = clustering
            # weights.clear()
            # for point in design_points:
            #     weights[point] = 1 / len(design_points)

            denominator = 0
            gradients = dict()
            for idx in range(len(design_points)):
                gradients[design_points[idx]] = gradient_func(design_points[idx])
                max_gradient = max(gradients[design_points[idx]], max_gradient)
                denominator += weights[design_points[idx]] * gradients[design_points[idx]]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            for idx in range(len(design_points)):
                weights[design_points[idx]] *= gradients[design_points[idx]] / denominator
            idx = 0
            while idx < len(design_points):
                if weights[design_points[idx]] <= 1e-14:
                    weights.pop(design_points[idx])
                    design_points.pop(idx)
                else:
                    idx += 1
            model_util.cal_inf_mat_w(design_points, weights)
            print(len(design_points))
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def plot_eq(self):
        target = self.model_util.gradient_target()
        plt.xticks([])
        plt.xlim(0, len(self.candidate_set) - 1)
        plt.ylim(-target, 1)
        x = [i for i in range(len(self.candidate_set))]
        y = [self.model_util.gradient_func(point) - target for point in self.candidate_set]
        plt.xlabel(r'$\chi$')
        plt.ylabel(r'$\phi(\mathbf{x},\xi)$')
        plt.plot(x, y, c='0.3')
        plt.show()
    # def gex(self):
    # def test(self):
