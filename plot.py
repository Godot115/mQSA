# -*- coding: utf-8 -*-
# @Time    : 2/27/23 23:59
# @Author  : godot
# @FileName: plot.py
# @Project : mQSA
# @Software: PyCharm


import pandas as pd

# 每个算法在不同标称值下的最优值
# 每个算法在不同标称值下的平均值
# 每个算法在不同标称值下，最优粒子的平均收敛历史

# Fig 1
# read data from total_vs.csv
# df = pd.read_csv("bests_theta_algorithm.csv")
# # plot the best fitness history for each theta and algorithm
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# algodict = {"mulQSA": "proposed algorithm",
#             "SADE": "SaDE",
#             "JADE": "JADE",
#             "BaseDE": "DE/rand/2/bin",
#             "BaseGA": "GA",
#             "OriginalQSA": "QSO"}
# # 使用不同线段区分，使用黑白颜色
# # plot four figures in two row two column
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))
# axs = axs.flatten()
#
# for i, theta in enumerate(df.theta.unique()):
#     for algorithm in df.algorithm.unique():
#         best = df[(df.theta == theta) & (df.algorithm == algorithm)]
#         # delete algorithm,theta,seed,best_fitness from best
#         best.drop(["algorithm", "theta", "seed", "best_fitness"], axis=1, inplace=True)
#         best = best.values.tolist()[0]
#         axs[i].plot(best, label=algodict[algorithm])
#
#         # Set the title and legend for the current subplot
#     axs[i].set_title(f"Nominal parameters is set to $\\theta_{i+1}$")
#     axs[i].legend(fontsize=10)
#
#     # Set the title for the entire figure and save it
# plt.savefig("fitness_vs_algorithm.png")

# Fig 2
# read data from logistic_means_theta_algorithm.csv and stds_theta_algorithm.csv
# means = pd.read_csv("logistic_means_theta_algorithm.csv")
# sems = pd.read_csv("sems_theta_algorithm.csv")
# # plot the mean best fitness for each theta and algorithm , with std, in a bar figure
# import matplotlib.pyplot as plt
# algodict = {"mulQSA": "proposed algorithm",
#             "SADE": "SaDE",
#             "JADE": "JADE",
#             "BaseDE": "DE/rand/2/bin",
#             "BaseGA": "GA",
#             "OriginalQSA": "QSO"}
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))
# axs = axs.flatten()
#
# for i, theta in enumerate(means.theta.unique()):
#     for algorithm in means.algorithm.unique():
#         mean = means[(means.theta == theta) & (means.algorithm == algorithm)]
#         sem = sems[(sems.theta == theta) & (sems.algorithm == algorithm)]
#         # delete algorithm,theta,seed,best_fitness from mean
#         mean = mean['best_fitness']
#         sem = sem['sem']
#         axs[i].bar(algorithm, mean, yerr=sem, label=algodict[algorithm])
#         axs[i].set_title(f"Nominal parameters is set to $\\theta_{i+1}$")
#         axs[i].legend(fontsize=10)
#
# plt.savefig("mean_fitness_vs_algorithm.png")

algorithm_dict = {
    "mulQSA": "proposed algorithm",
    "SADE": "SaDE",
    "BaseDE": "DE",
    "BaseGA": "GA",
    "OriginalQSA": "QSO",
    "OriginalPSO": "PSO"
}
marker_dict = {
    "mulQSA": "o",
    "SADE": "v",
    "BaseDE": "s",
    "BaseGA": "p",
    "OriginalQSA": "1",
    "OriginalPSO": "X"
}

line_dict = {
    "mulQSA": "solid",
    "SADE": "dotted",
    "BaseDE": 'dashed',
    "BaseGA": 'dashdot',
    "OriginalQSA": (0, (5, 5)),
    "OriginalPSO": (0, (3, 1, 1, 1, 1, 1))
}

means = pd.read_csv("poisson_means_theta_algorithm.csv")
import matplotlib.pyplot as plt

plt.figure(dpi=300)
# set colors as gist_gray
ax = plt.gca()  # gca:get current axis得到当前轴
# 设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
for i, theta in enumerate(means.theta.unique()):
    for algorithm in means.algorithm.unique():
        mean = means[(means.theta == theta) & (means.algorithm == algorithm)]
        # delete algorithm,theta,seed,best_fitness from mean
        mean = mean.drop(["algorithm", "theta", "best_fitness"], axis=1)
        mean = mean.values.tolist()[0]
        plt.plot(mean, label=algorithm_dict[algorithm], marker=marker_dict[algorithm], markevery=40, markersize=3,
                 linestyle=line_dict[algorithm])
    plt.xlabel('Iteration')
    plt.ylabel('log-determinant of Fisher Information Matrix')
    plt.legend(fontsize=10)
    plt.savefig(f"mean_fitness_vs_algorithm_theta_{i + 1}_poisson.png")
    plt.cla()
