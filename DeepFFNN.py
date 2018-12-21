import numpy as np
import random
import time
import math
from NeuralNetwork import NeuralNetwork

#-------------------------------------------------------------------------------
# DEFINE A DEEP NEURAL NETWORK CLASS
# WITH THE ARCHITECTURE WE REQUIRE
# AND METHODS THAT MAKE IT AMENABLE TO BAYESIAN ML PROCESSES
#-------------------------------------------------------------------------------
class DeepFFNN(NeuralNetwork):

    def __init__(self, input, hidden, output, max_depth, output_act, eval_metric):

        self.hidden = hidden
        self.max_depth = max_depth
        NeuralNetwork.__init__(self, input, output, output_act, eval_metric)

        self.w_size = self.get_weight_vector_length()

        self.initialise_cache()

        # WEIGHTS FROM INPUT TO FIRST HIDDEN LAYER 
        self.W1 = np.random.randn(self.input, self.hidden) / np.sqrt(self.input)
        self.B1 = np.random.randn(1, self.hidden) / np.sqrt(self.hidden)

        # WEIGHTS FROM LAST HIDDEN LAYER TO OUTPUT LAYER  
        self.W2 = np.random.randn(self.hidden, self.output) / np.sqrt(self.hidden)
        self.B2 = np.random.randn(1, self.output) / np.sqrt(self.hidden)  

        self.out = np.zeros((1, self.output))  # output layer for base model

        # NOW LETS CREATE ALL OF THE HIDDEN LAYERS
        self.h_weights = []
        self.h_biases = []
        self.h_out = []
        for layer in range(self.max_depth):
            self.h_weights.append(np.random.randn(self.hidden, self.hidden) / np.sqrt(self.hidden))
            self.h_biases.append(np.random.randn(1, self.hidden) / np.sqrt(self.hidden)  )
            self.h_out.append(np.zeros((1, self.hidden)) )

        self.final_out = np.zeros((1, self.output))  # Final output for the model

    ######################################################################
    # PRINT THE ARCHITECTURE
    ######################################################################
    def print(self):
        print("Bayesian Deep Feed Forward Neural Network")
        print("Input Nodes:", self.input)
        print("Hidden Nodes:", self.hidden)
        print("Hidden Layers:", self.max_depth)
        print("Output Nodes:", self.output)



    ######################################################################
    # PASS DATA X THROUGH THE NETWORK TO PRODUCE AN OUTPUT
    ######################################################################
    def forward_pass(self, X):
        # INPUT LAYER FIRST
        z1 = X.dot(self.W1) - self.B1
        # OUTPUT OF THE FIRST HIDDEN NODES
        tempout = self.sigmoid(z1)  
        # NOW THE ADDITIONAL HIDDEN LAYERS
        for layer in range(self.max_depth):
            tempz1 = tempout.dot(self.h_weights[layer]) - self.h_biases[layer]
            tempout = self.sigmoid(tempz1)
            self.h_out[layer] = tempout
        z2 = tempout.dot(self.W2) - self.B2
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
        internal_layer_wts = self.hidden * self.hidden

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

        # ALL OF THE ADDITIONAL HIDDEN LAYER WEIGHTS AT THE END OF THE VECTOR 
        for layer in range(self.max_depth):
            w_layer_temp = w[start_index: start_index + internal_layer_wts]
            self.h_weights[layer] = np.reshape(w_layer_temp, (self.hidden, self.hidden))
            start_index = start_index + internal_layer_wts
            self.h_biases[layer] = w[start_index:start_index + self.hidden]
            start_index = start_index + self.hidden


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
    # THEN PASS THE SET OF DATA THROUGH THE NETWORK AND
    # FINALLY CALCULATING THE RMSE
    ######################################################################
    def evaluate_proposal(self, data, w):
        self.decode(w)
        fx = self.process_data(data)
        y = data[:, self.input]
        rmse = self.rmse(fx, y)
        return [fx, rmse]


    ######################################################################
    # LOG LIKELIHOOD
    # CALCULATED GIVEN
    # - A PROPOSED SET OF WEIGHTS
    # - A DATA SET
    # - AND THE PARAMETERS FOR THE ERROR DISTRIBUTION
    ######################################################################
    def log_likelihood(self, data, w, tausq):
        y = data[:, self.input]
        [fx, rmse] = self.evaluate_proposal(data, w)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return np.sum(loss)


    ######################################################################
    # LOG PRIOR
    ######################################################################
    def log_prior(self, w, tausq):
        h = self.hidden  # number hidden neurons in each layer.
        tot_h = h * (self.max_depth+1)
        d = self.input  # number input neurons
        part1 = -1 * ((d * tot_h + tot_h + 2) / 2) * np.log(self.sigma_squared)
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
        internal_layer_wts = self.hidden * self.hidden
        start_index = start_index + input_layer_wts
        start_index = start_index + self.hidden
        start_index = start_index + output_layer_wts
        start_index = start_index + self.output
        for layer in range(self.max_depth):
            start_index = start_index + internal_layer_wts
            start_index = start_index + self.hidden
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

    ######################################################################
    # GET THE WEIGHT VECTOR
    ######################################################################
    def get_weight_vector(self):
        mytemp = [get_weight_vector_length()]
        # TODO
        return mytemp


