import numpy as np
import random
import time
import math
from LangevinNeuralNetwork import LangevinNeuralNetwork

#-------------------------------------------------------------------------------
# A Langevin Bayesian Neural Network 
# 
# Contains a rnage of methods that make a neural network learner amenable to
# learning a set of weightings using a MCMC process with Langevin dynamics.
#-------------------------------------------------------------------------------
class LangevinFFNN(LangevinNeuralNetwork):

    def __init__(self, input, hidden, output, output_act, eval_metric):

        self.hidden = hidden
        self.sgd_runs = 1
        LangevinNeuralNetwork.__init__(self, input, output, output_act, eval_metric) 

        self.w_size = self.get_weight_vector_length()
        # for Equation 9 in Ref [Chandra_ICONIP2017]
        self.sigma_diagmat = np.zeros((self.w_size, self.w_size))  
        np.fill_diagonal(self.sigma_diagmat, self.step_w)

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
        print("Bayesian Langevin FEED FORWARD Neural Network")
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
    # BATCH UPDATE FUNCTIONS FOR STOCHASTIC GRADIENT DESCENT
    # WE STORE THE WEIGHT AND BIAS UPDATES UNTIL THE BATCH IS FINISHED
    # THEN APPLY THEM
    ######################################################################
    def reset_batch_update(self):
        self.W2_batch_update = self.W2.copy()
        self.B2_batch_update = self.B2.copy()
        self.W1_batch_update = self.W1.copy()
        self.B1_batch_update = self.B1.copy()

    def apply_batch_update(self):
        self.W2 = self.W2_batch_update
        self.B2 = self.B2_batch_update
        self.W1 = self.W1_batch_update
        self.B1 = self.B1_batch_update

    ######################################################################
    # RUN THE ERROR BACK THROUGH THE NETWORK TO CALCULATE THE CHANGES TO
    # ALL PARAMETERS.
    # NOTE - THIS IS CALLED AFTER THE forward_pass
    #      - YOU NEED TO RUN reset_batch_update BEFORE STARTING THE BATCH
    ######################################################################
    def backward_pass(self, Input, desired):
        out_delta = (desired - self.final_out) * (self.final_out * (1 - self.final_out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        for x in range(0, self.hidden):
            for y in range(0, self.output):
                self.W2_batch_update[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in range(0, self.output):
            self.B2_batch_update[y] += -1 * self.lrate * out_delta[y]

        for x in range(0, self.input):
            for y in range(0, self.hidden):
                self.W1_batch_update[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in range(0, self.hidden):
            self.B1_batch_update[y] += -1 * self.lrate * hid_delta[y]


    ######################################################################
    # RETURN AN UPDATED WEIGHT VECTOR USING GRADIENT DESCENT
    # BackPropagation with SGD
    ######################################################################
    def langevin_gradient_update(self, data, w):  
        
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        self.reset_batch_update()

        Input = np.zeros((1, self.input))  # temp hold input
        Desired = np.zeros((1, self.output))
        fx = np.zeros(size)

        for i in range(0, self.sgd_runs):
            for j in range(0, size):
                pat = j
                Input = data[pat, 0:self.input]
                Desired = data[pat, self.input:]
                self.forward_pass(Input)
                self.backward_pass(Input, Desired)
        self.apply_batch_update()
        w_updated = self.encode()

        return  w_updated


    ######################################################################
    # TAKE A SINGLE VECTOR OF FLOATING POINT NUMBERS AND USE IT TO 
    # SET THE VALUES OF ALL WEIGHTS AND BIASES
    ######################################################################
    def decode(self, w):
        input_layer_wts = self.input * self.hidden
        output_layer_wts = self.hidden * self.output

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
    # ENCODE THE WEIGHTS AND BIASES INTO A SINGLE VECTOR 
    ######################################################################
    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, self.B1, w2, self.B2])
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
        h = self.hidden  # number hidden neurons
        d = self.input  # number input neurons
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
        return self.calculate_langevin_metropolis_hastings_acceptance_probability(new_w, new_tausq, old_w, old_tausq, data)

    ######################################################################
    # GET THE WEIGHT VECTOR
    ######################################################################
    def get_weight_vector(self):
        mytemp = [get_weight_vector_length()]
        return mytemp


