#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def ensure_resultsdir(results_dir):
    print("testing for ", results_dir)
    directory = os.path.abspath(results_dir)
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
    num_samples = 10000  
    estimator = mcmc.MCMC(num_samples, traindata, testdata, neuralnet, results_path, eval_metric)  
    estimator.print()
    [pos_w, pos_tau, eval_train, eval_test, accept_ratio, test_preds_file] = estimator.sampler()

    print("\nTraining complete")

    burnin = 1000
    # PREVIOUSLY: 0.1 * num_samples  
    use_samples = 1000

    burn_x = []
    burn_y = []

    burnfile = results_path + "burnin.tsv"
    outburn = open(burnfile, 'w')
    outburn.write("Burnin\t" + eval_metric + "\r\n")    
    for i in range( int((num_samples-use_samples)/burnin)):
        burner = (i+1)*burnin
        endpoint = burner + use_samples
        eval_temp = np.mean(eval_test[int(burner):endpoint])
        burn_x.append(burner)
        burn_y.append( eval_temp )
        outburn.write("%f\t%f\r\n" % (burner, eval_temp) )

    outburn.close()

    burnin = num_samples - use_samples

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

    create_weight_boxplot( pos_w, results_path )

    create_test_forecast_bands(burnin, input, test_preds_file, testdata, results_path)


#################################################################################
# PLOT CONFIDENCE INTERVAL
#################################################################################
def plot_timeseries_confidence_intervals( burnin, eval_train, eval_test, results_path ):

    fx_train_final = eval_train[int(burnin):, ]
    fx_test_final = eval_test[int(burnin):, ]

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
def create_weight_boxplot( pos_w, results_path ):
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    ax.boxplot(pos_w)
    ax.set_xlabel('Weights & Biases')
    ax.set_ylabel('Posterior')

    plt.title("Boxplot of Posterior W (weights and biases)")
    plt.savefig(results_path + 'w_pos.png')
    plt.savefig(results_path + 'w_pos.svg', format='svg', dpi=600)
    plt.clf()

#################################################################################
# TURN THE TEST PREDICTIONS INTO MEAN PREDICTION AND A RANGE OF QUANTILE BANDS
# THEN CALCULATE SUMMARY STATISTICS ABOUT THE CALIBRATION OF THE MODEL 
#################################################################################
def create_test_forecast_bands(burnin, input, test_preds_file, testdata, results_path):
    # OPEN THE RESULTS FILE
    rez = np.loadtxt(test_preds_file)
    # CULL THE BURNIN
    tested = rez[int(burnin):, ]
    fx_mu = tested.mean(axis=0)
    fx_99 = np.percentile(tested, 99, axis=0)
    fx_95 = np.percentile(tested, 95, axis=0)
    fx_90 = np.percentile(tested, 90, axis=0)
    fx_80 = np.percentile(tested, 80, axis=0)
    fx_20 = np.percentile(tested, 20, axis=0)
    fx_10 = np.percentile(tested, 10, axis=0)
    fx_5 = np.percentile(tested, 5, axis=0)
    fx_1 = np.percentile(tested, 1, axis=0)
    y_test = testdata[:, input]
    data = {'y':y_test, 'qrt_1':fx_1, 'qrt_5':fx_5, 'qrt_10':fx_10, 'qrt_20':fx_20, 'mu': fx_mu, 
                        'qrt_80':fx_80, 'qrt_90':fx_90, 'qrt_95':fx_95, 'qrt_99':fx_99 }
    df = pd.DataFrame(data)
    df.to_csv(results_path + 'testdata_prediction_intervals.csv', index=False)

    df["in_98_window"] = np.where( (df['y']>df['qrt_1']) & (df['y']<df['qrt_99']), 1, 0 )
    df["in_90_window"] = np.where( (df['y']>df['qrt_5']) & (df['y']<df['qrt_95']), 1, 0 )
    df["in_80_window"] = np.where( (df['y']>df['qrt_10']) & (df['y']<df['qrt_90']), 1, 0 )
    df["in_60_window"] = np.where( (df['y']>df['qrt_20']) & (df['y']<df['qrt_80']), 1, 0 )
    df["window_size_98"] =  df['qrt_99'] - df['qrt_1']
    df["window_size_90"] =  df['qrt_95'] - df['qrt_5']
    df["window_size_80"] =  df['qrt_90'] - df['qrt_10']
    df["window_size_60"] =  df['qrt_80'] - df['qrt_20']

    df["base_error"] =  fx_mu - y_test
    df["abs_error"] =  abs(df["base_error"])
    
    in_98_window = df["in_98_window"].mean()
    in_90_window = df["in_90_window"].mean()
    in_80_window = df["in_80_window"].mean()
    in_60_window = df["in_60_window"].mean()
    max_window_size_98 = df["window_size_98"].max() 
    mean_window_size_98 = df["window_size_98"].mean() 
    min_window_size_98 = df["window_size_98"].min() 

    max_window_size_90 = df["window_size_90"].max() 
    mean_window_size_90 = df["window_size_90"].mean() 
    min_window_size_90 = df["window_size_90"].min() 
    max_window_size_80 = df["window_size_80"].max() 
    mean_window_size_80 = df["window_size_80"].mean() 
    min_window_size_80 = df["window_size_80"].min()
 
    max_window_size_60 = df["window_size_60"].max() 
    mean_window_size_60 = df["window_size_60"].mean() 
    min_window_size_60 = df["window_size_60"].min() 

    sum_data = { 'window': [98,90,80,60],
                 'calibration':[in_98_window,in_90_window,in_80_window,in_60_window],
                 'min_size':[min_window_size_98,min_window_size_90,min_window_size_80,min_window_size_60],
                 'mean_size':[mean_window_size_98,mean_window_size_90,mean_window_size_80,mean_window_size_60],
                 'max_size':[max_window_size_98,max_window_size_90,max_window_size_80,max_window_size_60] }
    sum_df = pd.DataFrame(sum_data)
    sum_df.to_csv(results_path + 'testdata_calibration.csv', index=False)

if __name__ == "__main__": main()

