import hashlib
import numpy as np
import random
import time
import math

#-------------------------------------------------------------------------------
# DEFINE A NEURAL NETWORK BASE CLASS - FOR OTHER MODELS TO EXTEND
#-------------------------------------------------------------------------------
class NeuralNetwork:

    # THESE PARAMETERS CONTROL THE RANDOM WALK
    # THE FIRST THE CHANGES TO THE NETWORK WEIGHTS
    step_w = 0.01;

    # THE SECOND THE VARIATION IN THE NOISE DISTRIBUTION
    step_eta = 0.01;

    # THESE VALUES CONTROL THE INVERSE GAMMA FUNCTION
    # WHICH IS WHAT WE ASSUME TAU SQUARED IS DRAWN FROM
    # THIS IS CHOSEN FOR PROPERTIES THAT COMPLIMENT WITH THE GAUSSIAN LIKELIHOOD FUNCTION
    # IS THERE A REFERENCE FOR THIS?
    nu_1 = 0
    nu_2 = 0

    # SIGMA SQUARED IS THE ASSUMED VARIANCE OF THE PRIOR DISTRIBUTION
    # OF ALL WEIGHTS AND BIASES IN THE NEURAL NETWORK
    sigma_squared = 25

    ######################################################################
    # CONSTRUCTOR
    ######################################################################
    def __init__(self, input, output, output_act, eval_metric):
        self.input = input
        self.output = output
        self.output_act = output_act
        self.eval_metric = eval_metric

        if output_act=="sigmoid":
           self.activation = self.sigmoid
        elif output_act=="tanh":
           self.activation = self.tanh
        elif output_act=="relu":
           self.activation = self.relu
        else :
           self.activation = self.linear

        if eval_metric=="MAE":
           self.eval = self.mae
        elif eval_metric=="MAPE":
           self.eval = self.mape
        elif eval_metric=="MASE":
           self.eval = self.mase
        elif eval_metric=="MASEa":
           self.eval = self.mase
        elif eval_metric=="MASEb":
           self.eval = self.maseb
        else :
           self.eval = self.rmse


    ######################################################################
    # LOCAL DEFINITION OF THE SIGMOID FUNCTION FOR CONVENIENCE
    ######################################################################
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    ######################################################################
    # LOCAL DEFINITION OF THE TANH FUNCTION FOR CONVENIENCE
    ######################################################################
    def tanh(self, xin):
        bounder = lambda t: max(-709,min(709,t))
        x = np.array([[bounder(xi) for xi in xin[0]]])
        ex = np.exp(x)
        eminx = np.exp(-x)
        return (ex - eminx)/(ex + eminx)

    ######################################################################
    # RELU
    ######################################################################
    def relu(self, xin):
        bounder = lambda t: max(0,t)
        x = np.array([[bounder(xi) for xi in xin[0]]])
        return x

    ######################################################################
    # LINEAR ACTIVATION
    ######################################################################
    def linear(self, x):
        return x 

    ######################################################################
    # RMSE - Root Mean Squared Error
    ######################################################################
    def rmse(self, predictions, targets, features):
        return np.sqrt(((predictions - targets) ** 2).mean())

    ######################################################################
    # Mean Absolute Error
    ######################################################################
    def mae(self, predictions, targets, features):
        return (np.abs(predictions - targets)).mean()

    ######################################################################
    # Mean Absolute Percentage Error (with correction for zero target) 
    ######################################################################
    def mape(self, predictions, targets, features):
        return (np.abs(predictions - targets)/(targets+0.0000001)).mean()

    ######################################################################
    # AUC - Area Under the Curve (Binary Classification Only)
    # TODO: Implement
    ######################################################################
    def auc(self, predictions, targets, features):
        #joined = 
        #sorted = 
        return (np.abs(predictions - targets)/(targets+0.0000001)).mean()


    ######################################################################
    # Mean Absolute Scaled Error
    # This metric make strong assumptions about the structure of the data
    # 1. We assume that the last of the presented features is the 
    #    NAIVE prediction. This could be the previous known value of 
    #    the entity we are predicting, or the SEASONAL NAIVE VALUE
    #    Either way it is up to you to prepare the data this way.
    # NOTE: Adding a small value to correct for instances when
    #       the base error is zero
    ############################################################################
    def mase(self, predictions, targets, features):
        naive_preds = features[:, features.shape[1]-1 ]
        naive_error = np.abs(naive_preds - targets)
        model_error = np.abs(predictions - targets)
        return model_error.sum() / ( naive_error.sum() + 0.0000001 )

    ######################################################################
    # Mean Absolute Scaled Error - (Time Series Only) Second Version
    # This metric make strong assumption about the test data
    # 1. That its order in the vector is the order in time
    # 2. That the appropriate naive model is the last target value
    #    preceding the current row
    #    (in other words we are predicting one time step in advance)
    # NOTE: Adding a small value to correct for instances when
    #       the naive error is zero
    ######################################################################
    def mase_version2(self, predictions, targets, features):
        naive_preds = targets[0:len(targets)-1]
        naive_targs = targets[1:len(targets)]
        naive_error = np.abs(naive_preds - naive_targs)
        model_error = np.abs(predictions - targets)
        factor = len(targets) / (len(targets) - 1)
        return model_error.sum() / (factor * naive_error.sum() + 0.000001)


    ######################################################################
    # INITIALISE THE CACHES USED FOR STORING VALUES USED IN MCMC PROCESS
    ######################################################################
    def initialise_cache(self):
        self.log_likelihood_cache = {}
        self.log_prior_cache = {}

    ######################################################################
    # TRANSFORM A WEIGHT VECTOR INTO A KEY THAT WILL BE USED IN THE CACHE
    ######################################################################
    def get_cache_key(self, w):
        return hashlib.md5(str(w).encode('utf-8')).hexdigest()

    ######################################################################
    # GET METHOD FOR LOG LIKELIHOOD THAT MAKES USE OF THE CACHE
    ######################################################################
    def get_log_likelihood(self, data, w, tausq):
        tempkey = self.get_cache_key(w)
        if tempkey in self.log_likelihood_cache:
            return self.log_likelihood_cache[tempkey]
        else:
            templl = self.log_likelihood(data, w, tausq)
            self.log_likelihood_cache[tempkey] = templl
            return templl

    ######################################################################
    # GET METHOD FOR LOG PRIOR THAT MAKES USE OF THE CACHE
    ######################################################################
    def get_log_prior(self, w, tausq):
        tempkey = self.get_cache_key(w)
        if tempkey in self.log_prior_cache:
            return self.log_prior_cache[tempkey]
        else:
            templp = self.log_prior(w, tausq)
            self.log_prior_cache[tempkey] = templp
            return templp


    ######################################################################
    # CALCULATE EVERYTHING NEEDED FOR A SINGLE MCMC STEP 
    # - PROPOSAL, PREDS, ERROR AND ACCEPTANCE PROBABILITY
    ###########################################################################################################
    def get_proposal_and_acceptance_probability(self, w, eta, tausq, traindata, testdata):
            w_proposal = self.get_proposal_weight_vector(w)
            [eta_pro, tau_pro] = self.get_proposal_tau(eta)
            [pred_train, rmsetrain] = self.evaluate_proposal(traindata, w_proposal)
            [pred_test, rmsetest] = self.evaluate_proposal(testdata, w_proposal)
            mh_prob = self.get_acceptance_probability(w_proposal, tau_pro, w, tausq, traindata)
            return [w_proposal, eta_pro, tau_pro, pred_train, rmsetrain, pred_test, rmsetest, mh_prob]

    ######################################################################
    # CALCULATE METROPOLIS HASTINGS ACCEPTANCE PROBABILITY - RANDOM WALK
    #----------------------------------------------------------------------------------------
    # THIS NEXT SECTION INVOLVES CALCULATING: Metropolis-Hastings Acceptance Probability
    # This is what will determine whether a given change to the model weights (a proposal)
    # is accepted or rejected
    # This will consist of the following components
    # 1) A ratio of the likelihoods (current and proposal)
    # 2) A ratio of the priors (current and proposal)
    ###########################################################################################################
    def calculate_metropolis_hastings_acceptance_probability(self, new_w, new_tausq, old_w, old_tausq, data ):
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


