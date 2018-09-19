#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import os
import FFNN as ffnn
import MCMC as mcmc
import DeepFFNN as deepffnn

#################################################################################
# TRAIN A BAYESIAN NEURAL NETWORK 
# PARAMETERS
# - INPUT NODES
# - HIDDEN NODES
# - OUTPUT NODES
# - MAXIMUM DEPTH
# - PATH TO TRAINING DATA
# - PATH TO TESTING DATA
# - PATH TO RESULTS
# - SEED (OPTIONAL)
#################################################################################
def main():
    if len(sys.argv) < 8:
        print("ERROR: MISSING ARGUMENTS")
        print_usage(sys.argv)
        exit(1)
    else:
        input = int(sys.argv[1])
        hidden = int(sys.argv[2])
        output = int(sys.argv[3])
        depth = int(sys.argv[4])
        train_path = sys.argv[5]
        test_path = sys.argv[6]
        results_path = sys.argv[7]
        if len(sys.argv) > 8:
            rand_seed = sys.argv[8]
        else:
            rand_seed = 0
        np.random.seed(rand_seed)

        train_model(input, hidden, output, depth, train_path, test_path, results_path)
        #test_model(input, hidden, output, depth, train_path, test_path, results_path)

#################################################################################
def print_usage(args):
    print("USAGE ")
    print(args[0], "<INPUT NODES> <HIDDEN NODES> <OUTPUT NODES> <DEPTH> <TRAIN> <TEST> <RESULTS DIR> (<SEED>)")
    print("NOTE")
    print("THE NUMBER OF COLUMNS IN THE TRAIN AND TEST DATA MUST BE EQUAL TO INPUT PLUS OUTPUT NODES.")


#################################################################################
# TEST THE NN
#################################################################################
def test_model(input, hidden, output, depth, train_path, test_path, results_path):

    traindata = np.loadtxt(train_path)
    testdata = np.loadtxt(test_path)
    neuralnet = ffnn.FFNN(input, hidden, output, output_act='sigmoid')
    neuralnet.print() 
    testsize = testdata.shape[0]
    y_test = testdata[:, neuralnet.input]
    w_size = neuralnet.get_weight_vector_length()
    print("Total number of weights: " + str(w_size))
    w = np.random.randn(w_size)
    pred_test = neuralnet.evaluate_proposal(testdata, w)
    print("Records:", len(pred_test))
    print(type(pred_test))
    print("Record 0:", pred_test[0])


#################################################################################
# CREATE RESULTS DIRECTORY IF NEEDED
#################################################################################
def ensure_resultsdir(resultsdir):
    print("testing for ", resultsdir)
    directory = os.path.dirname(resultsdir)
    if not os.path.exists(directory):
        print("Does not exist... creating")
        os.makedirs(directory)

#################################################################################
# TRAIN THE MODELS
#################################################################################
def train_model(input, hidden, output, depth, train_path, test_path, results_path):
    ensure_resultsdir(results_path)
    rezfile = results_path + "results.txt"
    outres = open(rezfile, 'w')
    traindata = np.loadtxt(train_path)
    testdata = np.loadtxt(test_path)

    if depth>0:
        neuralnet = deepffnn.DeepFFNN(input, hidden, output, depth, output_act='sigmoid')
    else:
        neuralnet = ffnn.FFNN(input, hidden, output, output_act='sigmoid')
    neuralnet.print()

    random.seed( time.time() )
    numSamples = 50000  
    estimator = mcmc.MCMC(numSamples, traindata, testdata, neuralnet, results_path)  
    estimator.print()
    [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = estimator.sampler()

    print("\nTraining complete")

    burnin = 1000
    # PREVIOUSLY: 0.1 * numSamples  
    usesamples = 10000

    burn_x = []
    burn_y = []

    burnfile = results_path + "burnin.tsv"
    outburn = open(burnfile, 'w')
    outburn.write("Burnin\tTestRMSE\r\n")    
    for i in range( int((numSamples-usesamples)/burnin)):
        burner = (i+1)*burnin
        endpoint = burner + usesamples
        rmse_temp = np.mean(rmse_test[int(burner):endpoint])
        burn_x.append(burner)
        burn_y.append( rmse_temp )
        outburn.write("%f\t%f\r\n" % (burner,rmse_temp) )

    outburn.close()

    burnin = numSamples - usesamples

    pos_w = pos_w[int(burnin):, ]
    pos_tau = pos_tau[int(burnin):, ]

    fx_train_final = fx_train[int(burnin):, ]
    fx_test_final = fx_test[int(burnin):, ]

    fx_mu = fx_test_final.mean(axis=0)
    fx_high = np.percentile(fx_test_final, 95, axis=0)
    fx_low = np.percentile(fx_test_final, 5, axis=0)

    fx_mu_tr = fx_train_final.mean(axis=0)
    fx_high_tr = np.percentile(fx_train_final, 95, axis=0)
    fx_low_tr = np.percentile(fx_train_final, 5, axis=0)
 
    rmse_tr = np.mean(rmse_train[int(burnin):])
    rmsetr_std = np.std(rmse_train[int(burnin):])
    rmse_tes = np.mean(rmse_test[int(burnin):])
    rmsetest_std = np.std(rmse_test[int(burnin):])
    outres.write("Train_RMSE\t%f\r\n" % rmse_tr)
    outres.write("Train_RMSE_Std\t%f\r\n" % rmsetr_std)
    outres.write("Test_RMSE\t%f\r\n" % rmse_tes)
    outres.write("Test_RMSE_Std\t%f\r\n" % rmsetest_std)
    outres.write("Accept Ratio\t%f\r\n" % accept_ratio)
    outres.close()

    ytestdata = testdata[:, input]
    ytraindata = traindata[:, input]

    plt.plot(x_test, ytestdata, label='actual')
    plt.plot(x_test, fx_mu, label='pred. (mean)')
    plt.plot(x_test, fx_low, label='pred.(5th percen.)')
    plt.plot(x_test, fx_high, label='pred.(95th percen.)')
    plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Test Data vs MCMC Uncertainty ")
    plt.savefig(results_path + 'mcmcrestest.png')
    plt.savefig(results_path + 'mcmcrestest.svg', format='svg', dpi=600)
    plt.clf()

    plt.plot(x_train, ytraindata, label='actual')
    plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
    plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
    plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
    plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Train Data vs MCMC Uncertainty ")
    plt.savefig(results_path + 'mcmcrestrain.png')
    plt.savefig(results_path + 'mcmcrestrain.svg', format='svg', dpi=600)
    plt.clf()

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    ax.boxplot(pos_w)
    ax.set_xlabel('[W1] [B1] [W2] [B2]')
    ax.set_ylabel('Posterior')

    plt.title("Boxplot of Posterior W (weights and biases)")
    plt.savefig(results_path + 'w_pos.png')
    plt.savefig(results_path + 'w_pos.svg', format='svg', dpi=600)
    plt.clf()


if __name__ == "__main__": main()
