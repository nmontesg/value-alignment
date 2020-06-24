# -*- coding: utf-8 -*-
"""
@date: 16/02/2020

@author: Nieves Montes Gómez

@description: Find the alignment of prisoner alpha, who follows a strategy, with respect to values equality and personal
gain, as a function of the cooperation probability of beta, who performs random actions based on his/her probability of
cooperation attribute.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from prisoner_dilemma_model import StrategyDilemma

plt.rcParams.update({'font.size': 30})

length = 10  # length of paths
paths = 10000  # number of paths
norm0 = [6, 0, 9, 3]  # classical PD

probabilities = np.linspace(start=0, stop=1, num=11)
strategies = ["TitforTat", "MostlyCooperate", "MostlyDefect"]
align_eq = np.zeros(shape=(len(strategies), len(probabilities)))  # equation 7, it's the same for both agents
align_gain_alpha = np.zeros(shape=(len(strategies), len(probabilities)))  # equation 10
align_gain_beta = np.zeros(shape=(len(strategies), len(probabilities)))


def pref_personal_gain(gain, pdt):
	pdt_sorted = sorted(pdt)
	if gain == pdt_sorted[0]:
		return -1
	elif gain == pdt_sorted[1]:
		return -1 / 3
	elif gain == pdt_sorted[2]:
		return 1 / 3
	return 1


# iterate over alpha's strategies
for i in range(len(strategies)):
	# iterate over beta's cooperation probabilities
	for j in range(len(probabilities)):
		algn_eq_current, algn_gain_alpha_current, algn_gain_beta_current = 0., 0., 0.
		algn_agg_or_current, algn_agg_and_current = 0., 0.
		for _ in range(paths):
			PD = StrategyDilemma(*norm0, alpha_strat=strategies[i], beta_actions=probabilities[j])
			for _ in range(length):
				PD.step()

			# extract data
			out = PD.data_collector.get_agent_vars_dataframe()
			df_alpha = out.xs("alpha", level="AgentID")
			df_beta = out.xs("beta", level="AgentID")
			alpha = list(df_alpha.Wealth)
			beta = list(df_beta.Wealth)

			# preference over equality
			pref_eq = [1 - 2 * abs(alpha[i] - beta[i]) / (alpha[i] + beta[i]) for i in range(1, len(alpha))]
			algn_eq_current += sum(pref_eq)

			# preference over personal gain for alpha
			pref_gain_alpha = [pref_personal_gain(alpha[i] - alpha[i - 1], norm0) for i in range(1, len(alpha))]
			algn_gain_alpha_current += sum(pref_gain_alpha)

			# preference over personal gain for beta
			pref_gain_beta = [pref_personal_gain(beta[i] - beta[i - 1], norm0) for i in range(1, len(beta))]
			algn_gain_beta_current += sum(pref_gain_beta)

		align_eq[i][j] = algn_eq_current / (paths * length)
		align_gain_alpha[i][j] = algn_gain_alpha_current / (paths * length)
		align_gain_beta[i][j] = algn_gain_beta_current / (paths * length)

# save results to binary files
norm_str = "".join(map(str, norm0))
with open("results/alpha_strat_algn_eq_" + norm_str + ".nparray", "wb") as file:
	pickle.dump(align_eq, file)
with open("results/alpha_strat_algn_gain_alpha_" + norm_str + ".nparray", "wb") as file:
	pickle.dump(align_gain_alpha, file)
with open("results/alpha_strat_algn_gain_beta_" + norm_str + ".nparray", "wb") as file:
	pickle.dump(align_gain_beta, file)

# plot alignment with respect to equality
align_eq = pickle.load(open("results/alpha_strat_algn_eq_" + norm_str + ".nparray", 'rb'))

plt.figure(figsize=(12, 10))
plt.plot(probabilities, align_eq[0], color='black', marker='o', markersize=10, linewidth=2.5, label='Tit-for-tat')
plt.plot(probabilities, align_eq[1], color='blue', marker='o', markersize=10, linewidth=2.5, label='Mostly cooperate')
plt.plot(probabilities, align_eq[2], color='red', marker='o', markersize=10, linewidth=2.5, label='Mostly defect')
plt.grid()
# plt.legend(loc="center left", bbox_to_anchor=(1., 0.5))
plt.xlabel(r"Cooperation probability of $\beta$")
plt.ylabel(r"$\mathsf{Algn}^{\alpha,\beta}_{equality}$", labelpad=0)
plt.savefig("plots/alpha_strat_algn_eq_no_legend.eps", format='eps', bbox_inches='tight')

# plot alignment with respect to personal gain
# plot according to alignment with respect to personal gain
align_gain_alpha = pickle.load(open("results/alpha_strat_algn_gain_alpha_" + norm_str + ".nparray", 'rb'))
align_gain_beta = pickle.load(open("results/alpha_strat_algn_gain_beta_" + norm_str + ".nparray", 'rb'))

plt.figure(figsize=(22, 10))
plt.subplot(1, 2, 1)
plt.plot(probabilities, align_gain_alpha[0], color='black', marker='o', markersize=10, linewidth=2.5, label="Tit-for-tat")
plt.plot(probabilities, align_gain_alpha[1], color='blue', marker='o', markersize=10, linewidth=2.5, label='Mostly cooperate')
plt.plot(probabilities, align_gain_alpha[2], color='red', marker='o', markersize=10, linewidth=2.5, label='Mostly defect')
plt.grid()
plt.ylim(-0.65, 0.65)
plt.xlabel(r"Cooperation probability of $\beta$")
plt.ylabel(r"$\mathsf{Algn}^{i}_{gain}$", labelpad=0)

plt.subplot(1, 2, 2)
plt.plot(probabilities, align_gain_beta[0], color='black', marker='o', markersize=10, linewidth=2.5, label='Tit-for-tat')
plt.plot(probabilities, align_gain_beta[1], color='blue', marker='o', markersize=10, linewidth=2.5, label='Mostly cooperate')
plt.plot(probabilities, align_gain_beta[2], color='red', marker='o', markersize=10, linewidth=2.5, label='Mostly defect')
plt.grid()
plt.ylim(-0.65, 0.65)
plt.xlabel(r"Cooperation probability of $\beta$")
plt.legend(loc="center left", bbox_to_anchor=(1., 0.5))
plt.savefig("plots/alpha_strat_algn_gain.eps", format='eps', bbox_inches='tight')
