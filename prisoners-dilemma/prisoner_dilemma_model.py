# -*- coding: utf-8 -*-
"""
@date: 11/02/2020

@author: Nieves Montes Gómez

@description: A model of the Prisoner's Dilemma repeated game.
It includes the following classes for agents:
	- RandomPrisoner: an agent that behaves by taking random actions according to its probability of cooperation.
	- StrategyPrisoner: an agent that behaves strategically, by first taking a random action and then proceeding according
	to some predefined strategy based on the previous actions.

And the following classes for models:
	- RandomDilemma: a prisoner dilemma model where both agents are RandomPrisoner objects.
	- StrategyDilemma: child class where one agent is a RandomPrisoner object and the other is a StrategyPrisoner.
"""

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from numpy.random import choice


class RandomPrisoner(Agent):
	"""
	A prisoner who can cooperate or defect, randomly, based on attribute probability of cooperation.
	"""

	def __init__(self, unique_id, model, prob_coop):
		super().__init__(unique_id, model)
		self.wealth = 0.
		self.prob = prob_coop  # probability of cooperation
		self.action = choice(['C', 'D'], p=[self.prob, 1 - self.prob])

	def step(self):
		"""
		The prisoner chooses a random action according to his/her probability
		of cooperation.
		"""
		self.action = choice(['C', 'D'], p=[self.prob, 1 - self.prob])


class StrategyPrisoner(Agent):
	"""
	A prisoner with a defined strategy, depending on the previous action
	of the other agent. The first action is random.
	"""
	def __init__(self, unique_id, model, strategy):
		super().__init__(unique_id, model)
		self.wealth = 0.
		self.action = choice(['C', 'D'], p=[0.5, 0.5])
		self.strategy = strategy

	def step(self):
		"""
		Choose an action depending on the actions of the previous round.
		"""
		beta_prev_action = [ag.action for ag in self.model.schedule.agents if ag.unique_id == "beta"][0]
		if self.strategy == "TitforTat":
			if beta_prev_action == 'C':
				self.action = 'C'
			else:
				self.action = 'D'
		elif self.strategy == "MostlyCooperate":
			if (beta_prev_action == 'D') and (self.action == 'D'):
				self.action = 'D'
			else:
				self.action = 'C'
		elif self.strategy == "MostlyDefect":
			if (beta_prev_action == 'C') and (self.action == 'C'):
				self.action = 'C'
			else:
				self.action = 'D'
		else:
			raise SyntaxError("Strategy not found")


class RandomDilemma(Model):
	"""
	A model with two agents, prisoners in a game where their actions are
	distributed according to each agent's attribute probability of cooperation.
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_actions, beta_actions):
		super().__init__()
		self.num_agents = 2
		self.a, self.b, self.c, self.d = pdt_a, pdt_b, pdt_c, pdt_d
		self.running = True

		# create agents. They will be activated one at a time in the order of add
		self.schedule = BaseScheduler(self)
		alpha = RandomPrisoner("alpha", self, prob_coop=alpha_actions)
		beta = RandomPrisoner("beta", self, prob_coop=beta_actions)
		self.schedule.add(alpha)
		self.schedule.add(beta)

		# collect wealth of agents
		self.data_collector = DataCollector(agent_reporters=dict(Wealth="wealth", Action="action"))

	def step(self):
		"""
		Advance the model by one step:
			- collect the data on individual wealth.
			- increment agent's wealth according to their actions.
		"""
		self.data_collector.collect(self)
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
		self.schedule.step()


class StrategyDilemma(RandomDilemma):
	"""
	Prisoner's Dilemma model, with two prisoners. alpha follows a strategy,
	beta performs random action according to his/her probability of cooperation
	attribute.
	"""
	def __init__(self, pdt_a, pdt_b, pdt_c, pdt_d, alpha_strat, beta_actions):
		super().__init__(pdt_a, pdt_b, pdt_c, pdt_d, 0., beta_actions)

		# replace random agent alpha by strategic agent
		alpha_index = [agent.unique_id for agent in self.schedule.agents].index("alpha")
		_ = self.schedule.agents.pop(alpha_index)
		alpha = StrategyPrisoner("alpha", self, strategy=alpha_strat)
		self.schedule.add(alpha)
