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
    step_w = 0.02;

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
    def __init__(self, input, output, output_act):
        self.input = input
        self.output = output

        if output_act=="sigmoid":
           self.output_act = self.sigmoid
        elif output_act=="tanh":
           self.output_act = self.tanh
        else :
           self.output_act = self.identity


    ######################################################################
    # LOCAL DEFINITION OF THE SIGMOID FUNCTION FOR CONVENIENCE
    ######################################################################
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    ######################################################################
    # LOCAL DEFINITION OF THE TANH FUNCTION FOR CONVENIENCE
    ######################################################################
    def tanh(self, x):
        ex = np.exp(x)
        eminx = np.exp(-x)
        return (ex - eminx)/(ex + eminx)

    ######################################################################
    # IDENTITY FUNCTION FOR CONSISTENCY
    ######################################################################
    def identity(self, x):
        return x 

    ######################################################################
    # RMSE - Root Mean Squared Error
    ######################################################################
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

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
        return str(w)



