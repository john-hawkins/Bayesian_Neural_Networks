import numpy as np
import random
import time
import math
from NeuralNetwork import NeuralNetwork

#-------------------------------------------------------------------------------
# A STANDARD SINGLE LAYER PERCEPTRON 
# WITH THE METHODS THAT MAKE IT AMENABLE TO BAYESIAN ML PROCESSES
#-------------------------------------------------------------------------------
class SLP(NeuralNetwork):

    def __init__(self, input, output, output_act, eval_metric):

        NeuralNetwork.__init__(self, input, output, output_act, eval_metric) 

        self.w_size = self.get_weight_vector_length()

        self.initialise_cache()

        self.W1 = np.random.randn(self.input, self.output) / np.sqrt(self.input)
        self.B1 = np.random.randn(1, self.output) / np.sqrt(self.output)  # bias first layer

        self.out = np.zeros((1, self.output))  # output layer for base model

        self.final_out = np.zeros((1, self.output))  # Final output for the model

    ######################################################################
    # PRINT THE ARCHITECTURE
    ######################################################################
    def print(self):
        print("Bayesian Single Layer Perceptron")
        print("Input Nodes:", self.input)
        print("Output Nodes:", self.output)
        print("Output Activation", self.output_act)
        print("Eval Metric:", self.eval_metric)


    ######################################################################
    # PASS DATA X THROUGH THE NETWORK TO PRODUCE AN OUTPUT
    ######################################################################
    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.out = self.activation(z1)
        self.final_out = self.out
        return self.final_out


    ######################################################################
    # TAKE A SINGLE VECTOR OF FLOATING POINT NUMBERS AND USE IT TO 
    # SET THE VALUES OF ALL WEIGHTS AND BIASES
    ######################################################################
    def decode(self, w):
        input_layer_wts = self.input * self.output

        start_index = 0
        w_layer1 = w[start_index:input_layer_wts]
        self.W1 = np.reshape(w_layer1, (self.input, self.output))
        start_index = start_index + input_layer_wts

        self.B1 = w[start_index:start_index + self.output]
        start_index = start_index + self.output


    ######################################################################
    # ENCODE THE WEIGHTS AND BIASES INTO A SINGLE VECTOR 
    ######################################################################
    def encode(self):
        w1 = self.W1.ravel()
        w = np.concatenate([w1, self.B1])
        return w

    ######################################################################
    # PROCESS DATA
    # RUN A NUMBER OF EXAMPLES THROUGH THE NETWORK AND RETURN PREDICTIONS
    ######################################################################
    def process_data(self, data): 
        size = data.shape[0]
        Input = np.zeros((1, self.input))  # temp hold input
        Desired = np.zeros((1, self.output))
        fx = np.zeros(size)
        for pat in range(0, size):
            Input[:] = data[pat, 0:self.input]
            Desired[:] = data[pat, self.input:]
            self.forward_pass(Input)
            fx[pat] = self.final_out
        return fx


    ######################################################################
    # EVALUATE PROPOSAL 
    # THIS METHOD NEEDS TO SET THE WEIGHT PARAMETERS
    # THEN PASS THE SET OF DATA THROUGH, COLLECTING THE OUTPUT FROM EACH
    # OF THE BOOSTED LAYERS, AND THE FINAL OUTPUT
    ######################################################################
    def evaluate_proposal(self, data, w): 
        self.decode(w)  
        fx = self.process_data(data)
        y = data[:, self.input]
        feats =  data[:, :self.input]
        metric = self.eval(fx, y, feats)
        return [fx, metric]

    ######################################################################
    # LOG LIKELIHOOD
    # CALCULATED GIVEN 
    # - A PROPOSED SET OF WEIGHTS
    # - A DATA SET 
    # - AND THE PARAMETERS FOR THE ERROR DISTRIBUTION
    ######################################################################
    def log_likelihood(self, data, w, tausq):
        y = data[:, self.input]
        [fx, metric] = self.evaluate_proposal(data, w)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return np.sum(loss)


    ######################################################################
    # LOG PRIOR
    ######################################################################
    def log_prior(self, w, tausq):
        d = self.input  # number input neurons
        part1 = -1 * (d/2) * np.log(self.sigma_squared)
        part2 = 1 / (2 * self.sigma_squared) * (sum(np.square(w)))
        logp = part1 - part2 - (1 + self.nu_1) * np.log(tausq) - (self.nu_2 / tausq)
        return logp


    ######################################################################
    # GET THE COMPLETE LENGTH OF THE ENCODED WEIGHT VECTOR
    ######################################################################
    def get_weight_vector_length(self):
        start_index = 0
        input_layer_wts = self.input * self.output
        start_index = start_index + input_layer_wts
        start_index = start_index + self.output
        return start_index


    ######################################################################
    # GET NEW PROPOSAL WEIGHT VECTOR BY MODIFYING AN EXISTING ONE
    ######################################################################
    def get_proposal_weight_vector(self, w):
        w_proposal = w + np.random.normal(0, self.step_w, self.w_size)
        return w_proposal

    ######################################################################
    # GET PROPOSAL TAU VALUE FOR ERROR DISTRIBUTION 
    ######################################################################
    def get_proposal_tau(self, eta):
        eta_pro = eta + np.random.normal(0, self.step_eta, 1)
        tau_pro = math.exp(eta_pro)
        return [eta_pro, tau_pro]


    ######################################################################
    # ACCEPTANCE PROBABILITY - METROPOLIS HASTINGS
    ######################################################################
    def get_acceptance_probability(self, new_w, new_tausq, old_w, old_tausq, data ):
        return self.calculate_metropolis_hastings_acceptance_probability(new_w, new_tausq, old_w, old_tausq, data)

