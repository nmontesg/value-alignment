ValueAlignment
==============

This repository contains all the code developed during the work on the Master's thesis on Value Alignment in Multiagent Systems. It is all written in Python 3. Some of the scripts make use of the library [mesa](https://mesa.readthedocs.io/en/master/), which provides agent-based computation capabilities. A ```requirements.txt``` file is included, which specifies all the libraries and versions installed in the environment.

The code repository is organized into one folder per chapter for chapters 2 through 4. All folders include the necessary scripts, plus an additional directory for all the plots produced (in Encapsulated PostScript format) and other folders where important results are saved in json or binary files (which can be loaded using ```pickle```).

he folders and files correspond to the following implementations:

* The ```prisoners-dilemma``` folder includes all files related to Chapter 2. It contains the following scripts:

    * The script ```prisoner_dilemma_model.py``` contains the classes that model a random and strategic agent in an iterated Prisoner's Dilemma game, as well as classes for the model incorporating different types of agents.

    * The script ```random_PD_align.py``` performs the computations and produces the plots for the alignment under random-action profiles.

    * The script ```alpha_strategy_align.py``` performs the computations and produces the plots for the alignment under heterogeneous profiles.

* The ```prisoners-dilemma-norms``` folder contains the script ```PD_model_norms.py```, which implements the model with all the norms considered in Chapter 3. It consists of one general class for the norm-free iterated Prisoner's Dilemma game, and one subclass for each of the norms implemented. It also contains functions to compute the alignment, and produced all the plots presented in the chapter.

* The ```tax-model``` folder contains all scripts related to Chapter 4:

    * The script ```tax_model.py``` implements a class for an individual agent in the society and another class for the simple societal model considered in the chapter.
    * The script ```fitness_functions.py``` contains functions to compute the alignment of any model with respect to the three alignment functions that are considered.
    * The script ```genetic_optimizer.py``` implements the Genetic Algorithm optimization through a single class, which contains all the necessary steps of genetic optimization as methods.
    * The script ```genetic_search.py``` is the one that actually executes the genetic search.
    * The script ```auxiliary.py``` performs complementary computations, such as those related to the cross-alignment and it produces all the plots related to the evolution of the optimal models.
    * The script ```shapley.py``` implements the function for the computation of the Shapley values given the optimal model's parameters and the baseline ones. It runs the function with all of the three optimal models.
