# -*- coding: utf-8 -*-
"""
@date: 13/02/2020

@author: Nieves Montes Gómez

@description: Find the Nash equilibrium of a repeated Prisoner's Dilemma game for players with equal and unequal
quantification of payoff. Each prisoner's strategy is characterized by their probability of cooperating.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from prisoner_dilemma_model import RandomDilemma

plt.rcParams.update({'font.size': 30, 'figure.figsize': (10, 9)})

length = 10  # length of paths
paths = 10000  # number of paths
norm0 = [6, 0, 9, 3]  # classical PD

probabilities = np.linspace(start=0, stop=1, num=11)
algn_eq = np.zeros(shape=(len(probabilities), len(probabilities)))
algn_gain = np.zeros(shape=(len(probabilities), len(probabilities)))
algn_agg = np.zeros(shape=(len(probabilities), len(probabilities)))


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
for i in range(11):
	# iterate over beta's strategies
	for j in range(11):
		algn_eq_current, algn_gain_current = 0., 0.
		algn_agg_current = 0.
		for _ in range(paths):
			PD = RandomDilemma(*norm0, alpha_actions=probabilities[i], beta_actions=probabilities[j])
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

			# preference over personal gain
			pref_gain = [pref_personal_gain(alpha[i] - alpha[i - 1], norm0) for i in range(1, len(alpha))]
			algn_gain_current += sum(pref_gain)

		algn_eq[i][j] = algn_eq_current / (paths*length)
		algn_gain[i][j] = algn_gain_current / (paths*length)

# save results to binary files
norm_str = "".join(map(str, norm0))
with open("prisoner_dilemma/results/algn_eq_" + norm_str + ".nparray", "wb+") as file:
	pickle.dump(algn_eq, file)
with open("prisoner_dilemma/results/algn_gain_" + norm_str + ".nparray", "wb") as file:
	pickle.dump(algn_gain, file)

# import and plot in heat map alignment with respect to equality
algn_eq = pickle.load(open("prisoner_dilemma/results/algn_eq_" + norm_str + ".nparray", 'rb'))

fig, ax = plt.subplots()
cax = ax.imshow(algn_eq, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
cbar = fig.colorbar(cax)
cbar.set_label(r'$\mathsf{Algn}_{equality}^{\alpha, \beta}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')
plt.savefig("prisoner_dilemma/plots/random_PD_align_eq.png")


# plot according to alignment with respect to personal gain
algn_gain_alpha = pickle.load(open("prisoner_dilemma/results/algn_gain_" + norm_str + ".nparray", 'rb'))
algn_gain_beta = algn_gain_alpha.T

plt.subplots(figsize=(22, 9))
plt.subplot(1, 2, 2)
plt.imshow(algn_gain_beta, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='auto', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(start=-1, stop=1, num=5))
cbar.set_label(r'$\mathsf{Algn}_{gain}^{i}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

plt.subplot(1, 2, 1)
plt.imshow(algn_gain_alpha, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')
plt.savefig("prisoner_dilemma/plots/random_PD_align_gain.png")
