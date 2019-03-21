#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import os
import SLP as slp
import FFNN as ffnn
import LangevinFFNN as lgvnffnn
import MCMC as mcmc
import DeepFFNN as deepffnn
import DeepGBFFNN as deepgbffnn

#################################################################################
# TRAIN A BAYESIAN NEURAL NETWORK 
# PARAMETERS
# - INPUT NODES
# - HIDDEN NODES
# - OUTPUT NODES
# - MAXIMUM DEPTH
# - NETWORK ARCHITECTURE
# - OUTPUT ACTIVATION
# - PATH TO TRAINING DATA
# - PATH TO TESTING DATA
# - PATH TO RESULTS
# - EVAL METRIC
# - SEED (OPTIONAL)
#################################################################################
def main():
    if len(sys.argv) < 11:
        print("ERROR: MISSING ARGUMENTS")
        print_usage(sys.argv)
        exit(1)
    else:
        input = int(sys.argv[1])
        hidden = int(sys.argv[2])
        output = int(sys.argv[3])
        depth = int(sys.argv[4])
        architecture = sys.argv[5]
        activation = sys.argv[6]
        train_path = sys.argv[7]
        test_path = sys.argv[8]
        results_path = sys.argv[9]
        eval_metric = sys.argv[10]
        if len(sys.argv) > 11:
            rand_seed = sys.argv[11]
        else:
            rand_seed = 0
        np.random.seed(rand_seed)

        train_model(input, hidden, output, depth, architecture, activation, train_path, test_path, results_path, eval_metric)

#################################################################################
def print_usage(args):
    print("USAGE ")
    print(args[0], "<INPUT NODES> <HIDDEN NODES> <OUTPUT NODES> <DEPTH> <ARCH> <ACTIVATION> <TRAIN> <TEST> <RESULTS DIR> <EVAL METRIC> (<SEED>)")
    print("Valid model architectures: SLP FFNN DeepFFNN LangevinFFNN ")
    print("Valid output activation functions: linear sigmoid tanh relu")
    print("Valid eval metrics: RMSE MAE MAPE MASEa MASEb")
    print("NOTE")
    print("THE NUMBER OF COLUMNS IN THE TRAIN AND TEST DATA MUST BE EQUAL TO INPUT PLUS OUTPUT NODES.")


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
def train_model(input, hidden, output, depth, architecture, activation, train_path, test_path, results_path, eval_metric):
    ensure_resultsdir(results_path)
    rezfile = results_path + "results.txt"
    outres = open(rezfile, 'w')
    traindata = np.loadtxt(train_path)
    testdata = np.loadtxt(test_path)

    if architecture == 'DeepGBFFNN':
        neuralnet = deepgbffnn.DeepGBFFNN(input, hidden, output, depth, 0.05, activation, eval_metric)
    elif architecture == 'DeepFFNN':
        neuralnet = deepffnn.DeepFFNN(input, hidden, output, depth, activation, eval_metric)
    elif architecture == 'LangevinFFNN':
        neuralnet = lgvnffnn.LangevinFFNN(input, hidden, output, activation, eval_metric)
    elif architecture == 'SLP':
        neuralnet = slp.SLP(input, output, activation, eval_metric)
    else:
        neuralnet = ffnn.FFNN(input, hidden, output, activation, eval_metric)

    neuralnet.print()

    random.seed( time.time() )
    numSamples = 50000  
    estimator = mcmc.MCMC(numSamples, traindata, testdata, neuralnet, results_path, eval_metric)  
    estimator.print()
    [pos_w, pos_tau, eval_train, eval_test, accept_ratio] = estimator.sampler()

    print("\nTraining complete")

    burnin = 1000
    # PREVIOUSLY: 0.1 * numSamples  
    usesamples = 10000

    burn_x = []
    burn_y = []

    burnfile = results_path + "burnin.tsv"
    outburn = open(burnfile, 'w')
    outburn.write("Burnin\t" + eval_metric + "\r\n")    
    for i in range( int((numSamples-usesamples)/burnin)):
        burner = (i+1)*burnin
        endpoint = burner + usesamples
        eval_temp = np.mean(eval_test[int(burner):endpoint])
        burn_x.append(burner)
        burn_y.append( eval_temp )
        outburn.write("%f\t%f\r\n" % (burner, eval_temp) )

    outburn.close()

    burnin = numSamples - usesamples

    pos_w = pos_w[int(burnin):, ]
 
    eval_tr = np.mean(eval_train[int(burnin):])
    evaltr_std = np.std(eval_train[int(burnin):])
    eval_tst = np.mean(eval_test[int(burnin):])
    evaltest_std = np.std(eval_test[int(burnin):])
    outres.write("Train " + eval_metric + "\t%f\r\n" % eval_tr)
    outres.write("Train " + eval_metric + " Std\t%f\r\n" % evaltr_std)
    outres.write("Test " + eval_metric + "\t%f\r\n" % eval_tst)
    outres.write("Test " + eval_metric + " Std\t%f\r\n" % evaltest_std)
    outres.write("Accept Ratio\t%f\r\n" % accept_ratio)
    outres.close()

    createWeightBoxPlot( pos_w, results_path )


#################################################################################
# SOME PLOTTTING FUNCTIONs
#################################################################################
def plotTimeSeriesConfidenceInterval( results_path ):

    fx_train_final = fx_train[int(burnin):, ]
    fx_test_final = fx_test[int(burnin):, ]

    fx_mu = fx_test_final.mean(axis=0)
    fx_high = np.percentile(fx_test_final, 95, axis=0)
    fx_low = np.percentile(fx_test_final, 5, axis=0)
    
    fx_mu_tr = fx_train_final.mean(axis=0)
    fx_high_tr = np.percentile(fx_train_final, 95, axis=0)
    fx_low_tr = np.percentile(fx_train_final, 5, axis=0)
    
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


#################################################################################
# SOME PLOTTTING FUNCTION 
#################################################################################
def createWeightBoxPlot( pos_w, results_path ):
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    ax.boxplot(pos_w)
    ax.set_xlabel('Weights & Biases')
    ax.set_ylabel('Posterior')

    plt.title("Boxplot of Posterior W (weights and biases)")
    plt.savefig(results_path + 'w_pos.png')
    plt.savefig(results_path + 'w_pos.svg', format='svg', dpi=600)
    plt.clf()


if __name__ == "__main__": main()
