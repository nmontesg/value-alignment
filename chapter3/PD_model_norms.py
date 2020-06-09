# -*- coding: utf-8 -*-
"""
@date: 03/06/2020

@author: Nieves Montes Gómez

@description: ...
"""
# TODO: comment code properly
import copy
import itertools
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 30})

from numpy.random import choice, uniform
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner


def pref_GI(model):
	"""Compute the preference with respect to gain given the current post-transition
	state."""
	x_alpha = model.schedule.agents[0].wealth
	x_beta = model.schedule.agents[1].wealth
	return 1-2*abs(x_alpha-x_beta)/max(x_alpha+x_beta, 1.E-5)


class Prisoner(Agent):
	"""
	An agent in the Iterated Prisoner's Dilemma game. Action to take is chosen
	randomly according to a fixed probability of cooperation.
	"""
	def __init__(self, unique_id, model, prob_coop):
		super().__init__(unique_id, model)
		self.wealth = 0.
		self.prob = prob_coop  # probability of cooperation
		# first choice of actions is irrelevant, just not to throw an error
		self.action = choice(['C', 'D'], p=[self.prob, 1 - self.prob])

	def step(self):
		"""
		The prisoner chooses a random action according to his/her probability
		of cooperation.
		"""
		self.action = choice(['C', 'D'], p=[self.prob, 1 - self.prob])


class RandomDilemma(Model):
	"""
	A model with two agents, prisoners in a game where their actions are
	distributed according to each agent's attribute probability of cooperation.
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy):
		super().__init__()
		self.num_agents = 2
		self.a, self.b, self.c, self.d = pdt_a, pdt_b, pdt_c, pdt_d
		self.running = True

		# create agents. They will be activated one at a time in the order of add
		self.schedule = BaseScheduler(self)
		alpha = Prisoner("alpha", self, prob_coop=alpha_actions)
		beta = Prisoner("beta", self, prob_coop=beta_actions)
		self.schedule.add(alpha)
		self.schedule.add(beta)

		# collect wealth of agents
		self.data_collector = DataCollector(
			model_reporters=dict(pref_GI=pref_GI),
			agent_reporters=dict(Wealth="wealth", Action="action"))
		self.data_collector.collect(self)

	def step(self):
		"""
		Advance the model by one step:
			- all agents take their actions.
			- increment agent's wealth according to their actions.
			- collect the data on individual wealth.
		"""
		self.schedule.step()
		if self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.a
			self.schedule.agents[1].wealth += self.a
		elif self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.b
			self.schedule.agents[1].wealth += self.c
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.c
			self.schedule.agents[1].wealth += self.b
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.d
			self.schedule.agents[1].wealth += self.d
		self.data_collector.collect(self)
		
		
class RandomDilemmaFixedTaxes(RandomDilemma):
	"""
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy, tax_rate):
		super().__init__(pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy)
		self.tax_rate = tax_rate
		
	def step(self):
		"""
		Advance the model by one step:
			- all agents take their actions.
			- increment agent's wealth according to their actions.
			- collect the data on individual wealth.
		"""
		self.schedule.step()
		
		if self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.a * (1 - self.tax_rate)
			self.schedule.agents[1].wealth += self.a * (1 - self.tax_rate)
		elif self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.b * (1 - self.tax_rate)
			self.schedule.agents[1].wealth += self.c * (1 - self.tax_rate)
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.c * (1 - self.tax_rate)
			self.schedule.agents[1].wealth += self.b * (1 - self.tax_rate)
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.d * (1 - self.tax_rate)
			self.schedule.agents[1].wealth += self.d * (1 - self.tax_rate)
			
		self.data_collector.collect(self)	
		
		
class RandomDilemmaIncrementalTaxes(RandomDilemma):
	"""
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy,
							tax_a, tax_b, tax_c, tax_d):
		super().__init__(pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy)
		self.tax_a, self.tax_b, self.tax_c, self.tax_d = tax_a, tax_b, tax_c, tax_d
		
	def step(self):
		"""
		Advance the model by one step:
			- all agents take their actions.
			- increment agent's wealth according to their actions.
			- collect the data on individual wealth.
		"""
		self.schedule.step()
		if self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.a - self.tax_a
			self.schedule.agents[1].wealth += self.a - self.tax_a
		elif self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.b - self.tax_b
			self.schedule.agents[1].wealth += self.c - self.tax_c
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.c - self.tax_c
			self.schedule.agents[1].wealth += self.b - self.tax_b
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.d - self.tax_d
			self.schedule.agents[1].wealth += self.d - self.tax_d
			
		self.data_collector.collect(self)	
		
		
class RandomDilemmaNDefections(RandomDilemma):
	"""
	Class for a random dilemma model where no more than n defections are allowed.
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy, n):
		super().__init__(pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy)
		self.n = n
		
		# add counter for defections for each agent
		for ag in self.schedule.agents:
			ag.counter = 0
		
	def step(self):
		self.schedule.step()
		
		# if agents defect add to the counter
		for ag in self.schedule.agents:
			if ag.action == 'D':
				ag.counter += 1
			# if agent exceeds defections, change 
			if ag.counter > self.n:
				ag.action = 'C'
				ag.counter = 0
				
		if self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.a
			self.schedule.agents[1].wealth += self.a
		elif self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.b
			self.schedule.agents[1].wealth += self.c
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.c
			self.schedule.agents[1].wealth += self.b
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.d
			self.schedule.agents[1].wealth += self.d
			
		self.data_collector.collect(self)	
		
		
class RandomDilemmaDoubleDefection(RandomDilemma):
	"""
	Class for a random dilemma model where double defections are not allowed.
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy):
		super().__init__(pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions, dummy)
		
	def step(self):
		self.schedule.step()
						
		if self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.a
			self.schedule.agents[1].wealth += self.a
		elif self.schedule.agents[0].action == 'C' and self.schedule.agents[1].action == 'D':
			self.schedule.agents[0].wealth += self.b
			self.schedule.agents[1].wealth += self.c
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'C':
			self.schedule.agents[0].wealth += self.c
			self.schedule.agents[1].wealth += self.b
		elif self.schedule.agents[0].action == 'D' and self.schedule.agents[1].action == 'D':
			if uniform() < 0.5:
				self.schedule.agents[0].wealth += self.b
				self.schedule.agents[1].wealth += self.c
			else:
				self.schedule.agents[0].wealth += self.c
				self.schedule.agents[1].wealth += self.b
			
		self.data_collector.collect(self)	
		
		
		
def alignment_equality(model_cls, model_params, length, paths):
	"""
	Compute alignment with respect to equality.
	"""
	batch = BatchRunner(
		model_cls=model_cls,
		fixed_parameters=model_params,
		variable_parameters={'dummy': [0]},
		iterations=paths,
		max_steps=length,
		model_reporters={"all_steps": lambda m: m.data_collector.get_model_vars_dataframe()},
		display_progress=True
	)
	batch.run_all()
	all_dfs = batch.get_model_vars_dataframe()
	algn = 0
	for _, df in all_dfs['all_steps'].iteritems():
		algn += df['pref_GI'][1:].mean()
	return algn/paths
	
		
def alignment_gain(model_cls, model_params, max_M, min_M, length, paths):
	"""
	Compute alignment with respect to personal gain.
	"""		
	batch = BatchRunner(
		model_cls=model_cls,
		fixed_parameters=model_params,
		variable_parameters={'dummy': [0]},
		iterations=paths,
		max_steps=length,
		model_reporters={"all_steps": lambda m: m.data_collector.get_agent_vars_dataframe()},
		display_progress=True
	)
	batch.run_all()
	all_dfs = batch.get_model_vars_dataframe()
	algn = 0
	for _, df in all_dfs['all_steps'].iteritems():
		df = df.xs('alpha', level='AgentID')
		gains = df['Wealth'].diff()[1:]
		pref_gain = (2*gains-max_M-min_M)/(max_M-min_M)
		algn += pref_gain.mean()
	return algn/paths



def alignment_array(model_cls, model_params_no_probs, alignment_function, algn_func_extra=None, length=10, paths=10000):
	probabilities = np.linspace(0, 1, 11)
	algn_array = np.zeros(shape=(len(probabilities), len(probabilities)))
	for prob_alpha, prob_beta in itertools.product(probabilities, repeat=2):
		i = int(prob_alpha*10)
		j = int(prob_beta*10)
		
		model_params = copy.deepcopy(model_params_no_probs)
		model_params['alpha_actions'] = prob_alpha
		model_params['beta_actions'] = prob_beta
		
		# compute alignment for those probabilities of cooperation
		function_call = dict(
			model_cls=model_cls,
			model_params=model_params,
			length=length,
			paths=paths
			)
		if algn_func_extra:
			function_call.update(algn_func_extra)
		algn = alignment_function(**function_call)
		
		# add alignment to the array
		algn_array[i][j] = algn
		
	return algn_array
	
	
		
length = 10  # length of paths
paths = 10000  # number of paths
norm0 = [6, 0, 9, 3]  # classical PD
pdt_a, pdt_b, pdt_c, pdt_d = norm0
fixed_tax_rate = 1/3
incremental_taxes = [3, 0, 5, 0]
tax_a, tax_b, tax_c, tax_d = incremental_taxes

default_model_params = dict(
	pdt_a = pdt_a,
	pdt_b = pdt_b,
	pdt_c = pdt_c,
	pdt_d = pdt_d
	)

fixed_tax_model_params = dict(
	pdt_a = pdt_a,
	pdt_b = pdt_b,
	pdt_c = pdt_c,
	pdt_d = pdt_d,
	tax_rate = fixed_tax_rate
	)

incremental_tax_model_params = dict(
	pdt_a = pdt_a,
	pdt_b = pdt_b,
	pdt_c = pdt_c,
	pdt_d = pdt_d,
	tax_a = tax_a,
	tax_b = tax_b,
	tax_c = tax_c,
	tax_d = tax_d
	)

n_defection_params = copy.deepcopy(default_model_params)
n_defection_params['n'] = 2

#%%

# compute arrays for alignment with respect to equality for tax norms
array_equality_default = alignment_array(RandomDilemma, default_model_params, alignment_equality)
with open("results/array_equality_default.nparray", "wb+") as file:
	pickle.dump(array_equality_default, file)
	
array_equality_fixed = alignment_array(RandomDilemmaFixedTaxes, fixed_tax_model_params, alignment_equality)
with open("results/array_equality_fixed.nparray", "wb+") as file:
	pickle.dump(array_equality_fixed, file)
	
array_equality_incremental = alignment_array(RandomDilemmaIncrementalTaxes, incremental_tax_model_params, alignment_equality)
with open("results/array_equality_incremental.nparray", "wb+") as file:
	pickle.dump(array_equality_incremental, file)

#%%


#%%

# plot alignment arrays for value equality
array_equality_fixed = pickle.load(open("results/array_equality_fixed.nparray", "rb"))
array_equality_incremental = pickle.load(open("results/array_equality_incremental.nparray", "rb"))

plt.subplots(figsize=(25, 10))
plt.subplot(1, 2, 2)
plt.imshow(array_equality_incremental, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='auto', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(start=-1, stop=1, num=5))
cbar.set_label(r'$\mathsf{Algn}_{equality}^{\alpha, \beta}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')

plt.subplot(1, 2, 1)
plt.imshow(array_equality_fixed, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

plt.savefig('plots/array_equality_taxes.eps', format='eps', bbox_inches='tight')

#%%


#%%

# compute alignment with respect to gain for tax norms
additional_params = dict(
	max_M=max(norm0),
	min_M=min(norm0)
	)

array_gain_default = alignment_array(RandomDilemma, default_model_params, alignment_gain, algn_func_extra=additional_params)
with open("results/array_gain_default.nparray", "wb+") as file:
	pickle.dump(array_gain_default, file)
	
array_gain_fixed = alignment_array(RandomDilemmaFixedTaxes, fixed_tax_model_params, alignment_gain, algn_func_extra=additional_params)
with open("results/array_gain_fixed.nparray", "wb+") as file:
	pickle.dump(array_gain_fixed, file)
	
array_gain_incremental = alignment_array(RandomDilemmaIncrementalTaxes, incremental_tax_model_params, alignment_gain, algn_func_extra=additional_params)
with open("results/array_gain_incremental.nparray", "wb+") as file:
	pickle.dump(array_gain_incremental, file)


#%%


#%%

# plot alignment arrays for value personal gain
array_gain_fixed = pickle.load(open("results/array_gain_fixed.nparray", "rb"))
array_gain_incremental = pickle.load(open("results/array_gain_incremental.nparray", "rb"))

plt.subplots(figsize=(25, 10))
plt.subplot(1, 2, 2)
plt.imshow(array_gain_incremental, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='auto', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(start=-1, stop=1, num=5))
cbar.set_label(r'$\mathsf{Algn}_{gain}^{\alpha}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')

plt.subplot(1, 2, 1)
plt.imshow(array_gain_fixed, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

plt.savefig('plots/array_gain_taxes.eps', format='eps', bbox_inches='tight')

#%%


#%%

# compute alignment with respect to equality for systems with norms limiting transitions
array_equality_two_defections = alignment_array(RandomDilemmaNDefections, n_defection_params, alignment_equality)
with open("results/array_equality_two_defections.nparray", "wb+") as file:
	pickle.dump(array_equality_two_defections, file)
	
array_equality_double_defection = alignment_array(RandomDilemmaDoubleDefection, default_model_params, alignment_equality)
with open("results/array_equality_double_defection.nparray", "wb+") as file:
	pickle.dump(array_equality_double_defection, file)

#%%
	
#%%
	
# plot alignment arrays for value equality
array_equality_two_defections = pickle.load(open("results/array_equality_two_defections.nparray", "rb"))
array_equality_double_defection = pickle.load(open("results/array_equality_double_defection.nparray", "rb"))

plt.subplots(figsize=(25, 10))
plt.subplot(1, 2, 2)
plt.imshow(array_equality_double_defection, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='auto', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(start=-1, stop=1, num=5))
cbar.set_label(r'$\mathsf{Algn}_{equality}^{\alpha, \beta}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')

plt.subplot(1, 2, 1)
plt.imshow(array_equality_two_defections, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

plt.savefig('plots/array_equality_bans.eps', format='eps', bbox_inches='tight')
	
#%%


#%%

array_equality_default = pickle.load(open("results/array_equality_default.nparray", "rb"))

array_equality_comparison = np.zeros(shape=(11, 11))

relative_algn_cases = {
	'1': 'default ~ two defections ~ double defection',
	'2': 'default ~ double defection > two defections',
	'3': 'default ~ two defections > double defection',
	'4': 'two defections > default > double defection'
	}

threshold = 0.1

for i, j in itertools.product(range(11), repeat=2):
	default_twodefections = array_equality_default[i][j] - array_equality_two_defections[i][j]
	default_doubledefection = array_equality_default[i][j] - array_equality_double_defection[i][j]
	twodefections_doubledefection = array_equality_two_defections[i][j] - array_equality_double_defection[i][j]
	
	if abs(default_twodefections) < threshold and abs(default_doubledefection) < threshold:
		array_equality_comparison[i][j] = 1
				
	elif abs(default_twodefections) < threshold and default_doubledefection > threshold:
		array_equality_comparison[i][j] = 2
		
	elif default_twodefections < -threshold and abs(default_doubledefection) < threshold:
		array_equality_comparison[i][j] = 3
		
	elif default_twodefections < -threshold and default_doubledefection > threshold:
		array_equality_comparison[i][j] = 4
	
	else:
		array_equality_comparison[i][j] = 5
	

plt.figure(figsize=(12, 10))	
im = plt.imshow(array_equality_comparison, origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal')
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

values = np.unique(array_equality_comparison.ravel())
colors = [im.cmap(im.norm(value)) for value in values]
labels = [
	r'default$\sim$two consec. def.$\sim$mutual def.',
	r'default$\sim$two consec. def.$\succ$mutual def.',
	r'two consec. def.$\succ$default$\sim$mutual def.',
	r'two consec. def.$\succ$default$\succ$mutual def.'
	]
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0. )
plt.grid()

plt.savefig('plots/array_equality_comparison_ban.eps', format='eps', bbox_inches='tight')

#%%


#%%

# compute alignment with respect to gain for systems with norms limiting transitions
array_gain_two_defections = alignment_array(RandomDilemmaNDefections, n_defection_params, alignment_gain, algn_func_extra=additional_params)
with open("results/array_gain_two_defections.nparray", "wb+") as file:
	pickle.dump(array_gain_two_defections, file) 
	
array_gain_double_defection = alignment_array(RandomDilemmaDoubleDefection, default_model_params, alignment_gain, algn_func_extra=additional_params)
with open("results/array_gain_double_defection.nparray", "wb+") as file:
	pickle.dump(array_gain_double_defection, file)

#%%


#%%

# plot alignment arrays for value personal gain
array_gain_two_defections = pickle.load(open("results/array_gain_two_defections.nparray", "rb"))
array_gain_double_defection = pickle.load(open("results/array_gain_double_defection.nparray", "rb"))

plt.subplots(figsize=(25, 10))
plt.subplot(1, 2, 2)
plt.imshow(array_gain_double_defection, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='auto', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(start=-1, stop=1, num=5))
cbar.set_label(r'$\mathsf{Algn}_{gain}^{\alpha}$', rotation=90)
plt.xlabel(r'Cooperation probability of $\beta$')

plt.subplot(1, 2, 1)
plt.imshow(array_gain_two_defections, cmap='binary', origin='lower', extent=[-0.05, 1.05, -0.05, 1.05], aspect='equal', vmin=-1, vmax=1)
plt.xticks([0, 0.5, 1])
plt.xlabel(r'Cooperation probability of $\beta$')
plt.ylabel(r'Cooperation probability of $\alpha$')

plt.savefig('plots/array_gain_bans.eps', format='eps', bbox_inches='tight')

#%%


