import hashlib
import numpy as np
import random
import time
import math
from NeuralNetwork import NeuralNetwork

#-------------------------------------------------------------------------------
# DEFINE A LANGEVIN NEURAL NETWORK BASE CLASS 
# WE EXTEND FROM THE STANDARD NEURAL NETWORK BUT CHANGE SOME OF THE MCMC FUNCTION
# DEFINTIONS AND SIGNATURES
#-------------------------------------------------------------------------------
class LangevinNeuralNetwork(NeuralNetwork):

    ######################################################################
    # CONSTRUCTOR
    ######################################################################
    def __init__(self, input, output, output_act):
        NeuralNetwork.__init__(self, input, output, output_act)


    ######################################################################
    # INITALISE THE DIAGONAL COVANRIANCE MATRIX FOR CALCULATING THE 
    # PROBABILITY OF PROPOSAL TRANSITIONS - NEEDED FOR GRADIENT DESCENT
    # MCMC 
    ######################################################################
    def initialise(self):
        self.w_size = self.get_weight_vector_length()
        # for Equation 9 in Ref [Chandra_ICONIP2017]
        self.sigma_diagmat = np.zeros((self.w_size, self.w_size))
        np.fill_diagonal(self.sigma_diagmat, self.step_w)


    #########################################################################################
    # CALCULATE LANGEVIN METROPOLIS HASTINGS ACCEPTANCE PROBABILITY - GRADIENT DESCENT + RANDOM WALK
    #----------------------------------------------------------------------------------------
    # THIS NEXT SECTION INVOLVES CALCULATING: Metropolis-Hastings Acceptance Probability for a model
    # with Langevin dynamics in the propsal generation process.
    # This will consist of the following components
    # 1) A ratio of the likelihoods (current and proposal)
    # 2) A ratio of the priors (current and proposal)
    # 3) The inverse ratio of the transition probabilities.
    ###########################################################################################################
    def calculate_metropolis_hastings_acceptance_probability(self, new_w, new_tausq, old_w, old_tausq, data ):

        w_prop_gd = self.langevin_gradient(data, new_w.copy())
        diff_prop =  np.log(multivariate_normal.pdf(old_w, w_prop_gd, self.sigma_diagmat)  - np.log(multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat)))

        new_log_prior = self.get_log_prior( new_w, new_tausq)
        new_log_likelihood = self.get_log_likelihood(data, new_w, new_tausq)
        old_log_prior = self.get_log_prior( old_w, old_tausq)
        old_log_likelihood = self.get_log_likelihood(data, old_w, old_tausq)
        diff_likelihood = new_log_likelihood - old_log_likelihood
        diff_prior = new_log_prior - old_log_prior
        logproduct = diff_likelihood + diff_prior
        if logproduct > 709:
            logproduct = 709
        difference = math.exp(logproduct)
        mh_prob = min(1, difference)
        return mh_prob



