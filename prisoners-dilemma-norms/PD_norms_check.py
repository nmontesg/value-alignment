# -*- coding: utf-8 -*-
"""
@date: 12/06/2020

@author: Nieves Montes Gómez
"""

from PD_model_norms import RandomDilemma, RandomDilemmaIncrementalTaxes
from PD_model_norms import default_model_params, incremental_tax_model_params


model = RandomDilemma(**default_model_params, alpha_actions=0.5, beta_actions=0.5)




