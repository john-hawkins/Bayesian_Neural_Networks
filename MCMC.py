import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import time
import math
import os

#-------------------------------------------------------------------------------
# DEFINE A MARKOV CHAIN MONTE CARLO CLASS
# SPECIFIC FOR THE NEURAL NETWORK CLASS HIERARCHY 
#-------------------------------------------------------------------------------
class MCMC:
    def __init__(self, samples, traindata, testdata, neuralnetwork, resultsdir, eval_metric):
        self.samples = samples  
        self.neuralnet = neuralnetwork
        self.traindata = traindata
        self.testdata = testdata
        self.resultsdir = resultsdir
        self.eval_metric = eval_metric
        self.ensure_resultsdir()

    def ensure_resultsdir(self):
        directory = os.path.dirname(self.resultsdir)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mae(self, predictions, targets):
        return (np.abs(predictions - targets)).mean()

    def mape(self, predictions, targets):
        return (np.abs(predictions - targets)/(targets+0.0000001)).mean()

    def reduce_data(self, data, incl):
        fltre = incl>0
        return data[fltre]

    def modify_included_data(self, incl):
        newincl = incl.copy()
        pos = random.choice(list(range(0, len(incl))))
        if newincl[pos]==0: 
           newincl[pos]=1
        else: 
           newincl[pos]=0
        return newincl

    def get_indecies(self, arr, val):
        return np.where(arr == val)[0]

    def modify_included_dataV2(self, incl):
        newincl = incl.copy()
        # DETERMINE WHETHER TO ADD OR REMOVE DATA
        action = random.uniform(0, 1)
        if (action< 0.5) :
           ind = self.get_indecies(incl, 0)
           newval = 1
        else: 
           ind = self.get_indecies(incl, 1)
           newval = 0
        if (len(ind) == 0):
           return newincl
        pos_ind = random.choice(list(range(0, len(ind))))
        pos = ind[pos_ind]
        newincl[pos]=newval
        return newincl

    ########################################################################################
    # 
    ########################################################################################
    def print(self):
        print("Training data:", len(self.traindata) )
        print("Testing data:", len(self.testdata) )


    ########################################################################################
    # 
    ########################################################################################
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()


    ########################################################################################
    # LOGGING UTILITIES
    ########################################################################################
    def start_log_file(self):
        self.logfile = self.resultsdir + "log.txt"
        self.outlog = open(self.logfile, 'w')

    def write_log_entry(self, iteration, message):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        self.outlog.write(st + "\t" + str(iteration) + "\t" + message + "\r\n")
        self.outlog.flush()

    def close_log(self):
        self.outlog.close()

    ########################################################################################
    # RUN THE MCMC SAMPLER
    ########################################################################################
    def sampler(self):
        self.start_log_file()
        self.printProgressBar(0, self.samples, prefix = 'Progress:', suffix = 'Complete', length = 50)
        self.write_log_entry(0, "Initialising...")

        # How many training and test points? 
        # shape[0] Returns the first dimension of the array
        # In this instance it is the number of discrete (x,y) combinations in the data set
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]

        # Samples is the number of samples we are going to take in the run of MCMC
        samples = self.samples

        # Copy the y values into an independent vector
        y_test = self.testdata[:, self.neuralnet.input]
        y_train = self.traindata[:, self.neuralnet.input]

        self.write_log_entry(0, "Training data size:" + str(y_train.size) )
        self.write_log_entry(0, "Testing data size:" + str(y_test.size) )

        # The total number of parameters for the neural network
        w_size = self.neuralnet.get_weight_vector_length()

	# Posterior distribution of all weights and bias over all samples
	# We will take 'samples' number of samples
	# and there will be a total of 'w_size' parameters in the model.
        # We collect this because it will hold the empirical data for our 
        # estimate of the posterior distribution. 
        pos_w = np.ones((samples, w_size))

        # TAU IS THE STANDARD DEVIATION OF THE ERROR IN THE DATA GENERATING FUNCTIONS
        # I.E. WE ASSUME THAT THE MODEL WILL BE TRYING TO LEARN SOME FUNCTION F(X)
	# AND THAT THE OBSERVED VALUES Y = F(X) + E
        # WE STORE THE POSTERIOR DISTRIBUTION OF TAU - AS GENERATED BY THE MCMC PROCESS 
        pos_tau = np.ones((samples, 1))

	# F(X) BUFFER - ALL NETWORK OUTPUTS WILL BE STORED HERE
        # REMOVED BECAUSE THE MEMEORY CONSTRAINT WAS TOO HIGH FOR LARGE DATASETS
        # TODO: CONSIDER WRITING DIRECTLY TO DISK EVERY SO OFTEN
        #fxtrain_samples = np.ones((samples, trainsize))  
        #fxtest_samples = np.ones((samples, testsize)) 

	# STORE EVAL METRIC FOR EVERY STEP AS THE MCMC PROGRESSES
        eval_train = np.zeros(samples)
        eval_test = np.zeros(samples)

	# WE INITIALISE THE WEIGHTS RANDOMLY  
        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)
        self.write_log_entry(0, "Weights Initialised" )

        [pred_train, etrain] = self.neuralnet.evaluate_proposal(self.traindata, w)
        [pred_test, etest] = self.neuralnet.evaluate_proposal(self.testdata, w)

        self.write_log_entry(0, "Initial Weights" + "\tTrain " + self.eval_metric + ": " + str(etrain) + "\tTest" + self.eval_metric + ":" + str(etest))

	# INITIAL VALUE OF TAU IS BASED ON THE ERROR OF THE INITIAL NETWORK ON TRAINING DATA
	# ETA - IS USED FOR DOING THE RANDOM WALK SO THAT WE CAN ADD OR SUBTRACT RANDOM VALUES
	#       SUPPORT OVER [-INF, INF]
	# IT WILL BE EXPONENTIATED TO GET tau_squared of the proposal WITH SUPPORT OVER [0, INF] 
        eta = np.log(np.var(pred_train - y_train))
        tausq = np.exp(eta)
 
        self.write_log_entry(0, "Initial Error Dist" + "\tEta " + str(eta) + "\tTau^2:" + str(tausq))

        likelihood = self.neuralnet.get_log_likelihood(self.traindata, w, tausq)

        self.write_log_entry(0, 'Initial Likelihood: ' + str(likelihood) )
        naccept = 0
        self.write_log_entry(0, 'Begin sampling using MCMC random walk')

        for i in range(samples - 1):
            self.printProgressBar(i + 1, samples, prefix = 'Progress:', suffix = 'Complete', length = 50)
 
            #w_proposal = self.neuralnet.get_proposal_weight_vector(w)
            #[eta_pro, tau_pro] = self.neuralnet.get_proposal_tau(eta)
            #[pred_train, rmsetrain] = self.neuralnet.evaluate_proposal(self.traindata, w_proposal)
            #[pred_test, rmsetest] = self.neuralnet.evaluate_proposal(self.testdata, w_proposal) 
            #mh_prob = self.neuralnet.get_acceptance_probability(w_proposal, tau_pro, w, tausq, self.traindata)
  
            [w_proposal, eta_pro, tau_pro, pred_train, etrain, pred_test, etest, mh_prob] = self.neuralnet.get_proposal_and_acceptance_probability(w, eta, tausq, self.traindata, self.testdata)

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                self.write_log_entry(i, "Proposal Accepted" + "\tTrain " + self.eval_metric + ": " + str(etrain) + "\tTest " + self.eval_metric + ": " + str(etest))
                naccept += 1
                w = w_proposal
                eta = eta_pro
                tausq = tau_pro

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                #fxtrain_samples[i + 1,] = pred_train
                #fxtest_samples[i + 1,] = pred_test
                eval_train[i + 1,] = etrain
                eval_test[i + 1,] = etest

            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                #fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                #fxtest_samples[i + 1,] = fxtest_samples[i,]
                eval_train[i + 1,] = eval_train[i,]
                eval_test[i + 1,] = eval_test[i,]
                self.write_log_entry( i,  "Proposal Rejected")

        self.write_log_entry(samples, str(naccept) + ' Accepted Samples')
        self.write_log_entry(samples, "Acceptance Rate:" + str(100 * naccept/(samples * 1.0)) + '%')
        accept_ratio = naccept / (samples * 1.0)
        self.close_log()

        #return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)
        return (pos_w, pos_tau, eval_train, eval_test, accept_ratio)


