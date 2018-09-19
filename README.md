Bayesian Neural Networks
========================
 
This goal of this project is to allow experimentation with different neural network structures
and variations on the MCMC sampling procedure.

[Inital code based on the MCMC FFN Project by Rohit](https://github.com/rohitash-chandra/MCMC_fnn_timeseries)


# Usage

The [RUN.sh Script](Run.sh) gives examples of how to execute several network architectures on a
range of datasets.

This script will execute the python command line program [train_bn_mcmc.py](train_bn_mcmc.py)
Which should be executed as follows

```
python train_bn_mcmc.py <INPUT NODES> <HIDDEN NODES> <OUTPUT NODES> <DEPTH> <TRAIN> <TEST> <RESULTS> (OPTIONAL:<RANDOM SEED>)
```

It expects to be given a training and testing data set, and it expects the data to be a CSV file
in which the first  <INPUT NODES> number of columns are the numerical input features for the model.
And the final <OUTPUT NODES> number of columns contain the target values.

If the  <DEPTH> is 0 then it will be a standard FFNN with only a single layer of hidden neural. 
Higher depth values add additional hidden layers

TODO: make this parameter a more intuitive direct description of the network architecture. 

CURRENT WORK: Extracting aspects of the metropolis hastings process that are specific to the
neural network architecture and embedding them in the ML Model class. So that the MCMC process
is general and abstract and we can run multiple architectures side-by-side.


