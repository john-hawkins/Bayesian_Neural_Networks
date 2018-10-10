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
python train_bn_mcmc.py <INPUT NODES> <HIDDEN NODES> <OUTPUT NODES> <DEPTH> <MODEL> <OUTPUT ACTIVATION> <TRAIN> <TEST> <RESULTS> (OPTIONAL:<RANDOM SEED>)
```

It expects to be given a training and testing data set, and it expects the data to be a CSV file
in which the first  <INPUT NODES> number of columns are the numerical input features for the model.
And the final <OUTPUT NODES> number of columns contain the target values.

The value of <MODEL> determines the overall network architecture, and <DEPTH> only applies if
it is a deep neural network. The value of <OUTPUT ACTIVATION> determines what the activation function
will be and you need to choose this depending on the distribution of your target value.


# TODO
 
* The above method of describing the neural network structure is cumbersome and inflexible. I plan to make this driven 
  by a single regular expression style syntax that describes the entire architecture.

# CURRENT WORK 

I am extracting aspects of the MCMC and Metropolis Hastings calculations that are specific to the
neural network architecture and embedding them in the specific neural network classes. 

This will make the overall MCMC class very abstract/general and I can then easily run multiple 
architectures side-by-side for comparison.




