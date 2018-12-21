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


    # THESE PARAMETERS CONTROL THE GRADIENT DESCENT PROCESS
    lrate = 0.01;

    ######################################################################
    # CONSTRUCTOR
    ######################################################################
    def __init__(self, input, output, output_act, eval_metric):
        NeuralNetwork.__init__(self, input, output, output_act, eval_metric)


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
    def calculate_metropolis_hastings_acceptance_probability( self, new_w, w_gd, new_tausq, old_w, old_tausq, data ):

        # WE FIRST NEED TO CALCULATE THE REVERSAL RATIO TO SATISFY DETAILED BALANCE
        # 
        # Calculate a weight vector derived using gradient descent from the proposal
        w_prop_gd = self.langevin_gradient_update(data, new_w.copy())

     
        # THIS WAS THE VERSION USED IN THE SIMULATIONS FOR THE ORIGINAL PAPER
        # diff_prop =  np.log(multivariate_normal.pdf(old_w, w_prop_gd, self.sigma_diagmat)  - np.log(multivariate_normal.pdf(new_w, w_gd, sigma_diagmat)))
        ### UPDATE TO FIX NUMERICAL ISSUE
        wc_delta = (old_w - w_prop_gd)
        wp_delta = (new_w - w_gd )
        sigma_sq = self.step_w
        first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
        second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq
        diff_prop =  first - second
        ### END UPDATE

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

    ######################################################################
    # GENERATE A PROPOSAL WEIGHT VECTOR USING GRADIENT DESCENT PLUS NOISE
    ######################################################################
    def get_proposal_weight_vector(self, data, w):
            w_gd = self.langevin_gradient_update(data, w)
            w_proposal = w_gd  + np.random.normal(0, self.step_w, self.w_size)
            return w_proposal

    ######################################################################
    # GENERATE A PROPOSAL WEIGHT VECTOR USING GRADIENT DESCENT PLUS NOISE
    # RETURN : Tuple with the proposal and the raw gradient descent derived 
    #           weights
    ######################################################################
    def get_proposal_weight_vector_and_gradient(self, data, w):
            w_gd = self.langevin_gradient_update(data, w)
            w_proposal = w_gd  + np.random.normal(0, self.step_w, self.w_size)
            return [w_proposal, w_gd]


    ######################################################################
    # CALCULATE EVERYTHING NEEDED FOR A SINGLE MCMC STEP
    # - PROPOSAL, PREDS, ERROR AND ACCEPTANCE PROBABILITY
    # We overwrite the definition in the Neural Network Base class because we need both the 
    # proposal generation and acceptance probability to be aware of the error gradient of model.
    ###########################################################################################################
    def get_proposal_and_acceptance_probability(self, w, eta, tausq, traindata, testdata):
            [w_proposal, w_gd] = self.get_proposal_weight_vector_and_gradient(traindata, w)
            [eta_pro, tau_pro] = self.get_proposal_tau(eta)
            [pred_train, rmsetrain] = self.evaluate_proposal(traindata, w_proposal)
            [pred_test, rmsetest] = self.evaluate_proposal(testdata, w_proposal)
            mh_prob = self.calculate_metropolis_hastings_acceptance_probability(w_proposal, w_gd, tau_pro, w, tausq, traindata)
            return [w_proposal, eta_pro, tau_pro, pred_train, rmsetrain, pred_test, rmsetest, mh_prob]


