import numpy as np
import random
import time
import math
from NeuralNetwork import NeuralNetwork

#-------------------------------------------------------------------------------
# A STANDARD FEED FORWARD NEURAL NETWORK CLASS
# WITH THE METHODS THAT MAKE IT AMENABLE TO BAYESIAN ML PROCESSES
#-------------------------------------------------------------------------------
class FFNN(NeuralNetwork):

    def __init__(self, input, hidden, output, output_act):
        self.hidden = hidden
        NeuralNetwork.__init__(self, input, output, output_act) 

        self.initialise_cache()

        self.W1 = np.random.randn(self.input, self.hidden) / np.sqrt(self.input)
        self.B1 = np.random.randn(1, self.hidden) / np.sqrt(self.hidden)  # bias first layer
        self.W2 = np.random.randn(self.hidden, self.output) / np.sqrt(self.hidden)
        self.B2 = np.random.randn(1, self.output) / np.sqrt(self.hidden)  # bias second layer

        self.hidout = np.zeros((1, self.hidden))  # output of first hidden layer
        self.out = np.zeros((1, self.output))  # output layer for base model

        self.final_out = np.zeros((1, self.output))  # Final output for the model

    ######################################################################
    # PRINT THE ARCHITECTURE
    ######################################################################
    def print(self):
        print("Bayesian FEED FORWARD Neural Network")
        print("Input Nodes:", self.input)
        print("Hidden Nodes:", self.hidden)
        print("Output Nodes:", self.output)


    ######################################################################
    # PASS DATA X THROUGH THE NETWORK TO PRODUCE AN OUTPUT
    ######################################################################
    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)
        self.final_out = self.out
        return self.final_out

    ######################################################################
    # TAKE A SINGLE VECTOR OF FLOATING POINT NUMBERS AND USE IT TO 
    # SET THE VALUES OF ALL WEIGHTS AND BIASES
    ######################################################################
    def decode(self, w):
        input_layer_wts = self.input * self.hidden
        output_layer_wts = self.hidden * self.output
        boost_layer_wts = self.hidden * self.hidden
        #print(input_layer_wts, output_layer_wts, boost_layer_wts)

        start_index = 0
        w_layer1 = w[start_index:input_layer_wts]
        self.W1 = np.reshape(w_layer1, (self.input, self.hidden))
        start_index = start_index + input_layer_wts

        self.B1 = w[start_index:start_index + self.hidden]
        start_index = start_index + self.hidden

        w_layer2 = w[start_index: start_index + output_layer_wts]
        self.W2 = np.reshape(w_layer2, (self.hidden, self.output))
        start_index = start_index + output_layer_wts

        self.B2 = w[start_index:start_index + self.output]
        start_index = start_index + self.output


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
        return self.process_data(data)

    def tempholder():
        size = data.shape[0]
        Input = np.zeros((1, self.input))  # temp hold input
        Desired = np.zeros((1, self.output))
        results = []
        for pat in range(0, size):
            Input[:] = data[pat, 0:self.input]
            Desired[:] = data[pat, self.input:]
            self.forward_pass(Input)
            results.append(self.out)
        return results


    ######################################################################
    # LOG LIKELIHOOD
    # CALCULATED GIVEN 
    # - A PROPOSED SET OF WEIGHTS
    # - A DATA SET 
    # - AND THE PARAMETERS FOR THE ERROR DISTRIBUTION
    ######################################################################
    def log_likelihood(self, data, w, tausq):
        y = data[:, self.input]
        fx = self.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]


    ######################################################################
    # LOG PRIOR
    ######################################################################
    def log_prior(self, w, tausq):
        h = self.hidden  # number hidden neurons
        d = self.output  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(self.sigma_squared)
        part2 = 1 / (2 * self.sigma_squared) * (sum(np.square(w)))
        logp = part1 - part2 - (1 + self.nu_1) * np.log(tausq) - (self.nu_2 / tausq)
        return logp


    ######################################################################
    # GET THE COMPLETE LENGTH OF THE ENCODED WEIGHT VECTOR
    ######################################################################
    def get_weight_vector_length(self):
        start_index = 0
        input_layer_wts = self.input * self.hidden
        output_layer_wts = self.hidden * self.output
        boost_layer_wts = self.hidden * self.hidden
        start_index = start_index + input_layer_wts
        start_index = start_index + self.hidden
        start_index = start_index + output_layer_wts
        start_index = start_index + self.output
        return start_index


    ######################################################################
    # GET NEW PROPOSAL WEIGHT VECTOR BY MODIFYING AN EXISTING ONE
    ######################################################################
    def get_proposal_weight_vector(self, w):
        w_proposal = w + np.random.normal(0, self.step_w, self.w_size)


    ######################################################################
    # GET THE TAU VALUE FOR ERROR DISTRIBUTION 
    ######################################################################
    def get_tau(self, eta):
        eta_pro = eta + np.random.normal(0, step_eta, 1)
        tau_pro = math.exp(eta_pro)
        return tau_pro


    ######################################################################
    # ACCEPTANCE PROBABILITY - METROPOLIS HASTINGS
    ######################################################################
    def get_acceptance_probability(self, new_weights, old_weights, new_tau, old_tau, data, ):
        new_key = self.get_cache_key(new_weights)
        old_key = self.get_cache_key(old_weights)
        #new_log_likelihood = 
        #old_log_likelihood = 
        w_proposal = w + np.random.normal(0, self.step_w, self.w_size)


    ######################################################################
    # GET THE WEIGHT VECTOR
    ######################################################################
    def get_weight_vector(self):
        mytemp = [get_weight_vector_length()]
        return mytemp


