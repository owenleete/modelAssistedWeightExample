# Example of model assisted weights method for reinforcement learning
This repo is intended to provide a publicly accessible simple example of a model assisted weights method for reinforcement learning.
The model assisted weight method is proposed in a research paper currently under review, an updated link will be provide when the paper is publicly available.

## Background
Reinforcement learning methods generally come in two types, model-based and model-free. 

Model-based methods assume that the future states of the MDP can be predicted using a transition dynamics model estimated from data. Model-based methods work well if the dynamics model is correctly specified, but can have high bias if the model is incorrect.

Model-free methods do not use a model for the transition dynamics, but instead model the state-value as a function of the state. For computational reasons, the state-value model is a linear combination of (possibly non-linear) basis functions of the state variables. These methods work well if the state-value model  is correct, but can perform poorly if the model is incorrect.

Model assisted weight is a method that we propose in the paper mentioned above that is designed to have good performance if either the transition dynamics model, or state-value model (but not necessarily both) is correct. If both models are correct, the model assisted weight method is asymptotically optimal in the sense of Godambe (1985). If the transition dynamics model is incorrect but the state value model is correct, the model assisted weight method is consistent for the correct answer. If the state-value model is incorrect but the transition dynamics model is correct, the model assisted weight method is consistent for the projection of the true state-values onto the linear span of the basis functions used in the state-value model.

## Files
#### DataObject.py
This file contains an object for generating storing and interacting with the simulated data. The data consist of N, T, state, action, probability, next state, and reward. N and T are the subject number and observation number respectively. State and next state are allowed to use multiple columns, but for this example they use only 1 column each. There are methods to generate the data, as well as extract each of the individual components of the data.

#### TransitonModel.py
This file contains an object for holding a transition dynamics model. There are methods to set or retrieve the transition dynamics model parameters. The primary method is "fitTransitionModel" which takes a DataObject as input and estimates the transition dynamics model parameters from the simulated data.

#### PolicyObject.py
This file contains an object for the action selection policy. The primary method is "getAction" which draws a random action based on the observed state and action selection policy parameters. There are also methods to calculate the probability of a given action based on the observed state.

#### FeatureModel.py
This file contains the feature space / basis function expansion of the observed state. This feature space is used to define the state-value model for V-learning. The primary methods are "getFeaturesCurrent" and "getFeaturesNext" which take a DataObject as input and return the features of the current states, and next states respectively.

#### VLearning.py
This file implements functions needed for V-learning. The primary method is "fitVlearning" which will return the parametes for the linear model that defines the state-value model. This file also contains the class "VLearningParameters" which holds hyper-parameters for fitting V-learning

#### modelAssisted.py
This file contains functions needed for the model-assisted weights approach. The primary method is "fitMA" which will return the parameters for the linear model that defines the state-value model

#### runExample.py
This file fits the V-learning and model-assisted weights approaches to simulated data. The output is the absolute error between the estimated parameters and the projection for each component of the state-value model.

##### Note:
The code used in the simulations for the paper use various efficiency improvements (numba, cplex, Go). This code has been simplified to run with minimal dependencies, but at a slower rate. The code is formatted to accommodate these efficiency improvements. This results in many helper/wrapper functions.


## Projection example
This setting demonstrates the ability of the model assisted weight method to find the projection. We define an MDP where the state-values could be accurately modeled with quadratic basis functions, but instead we use linear basis functions for the state-value model. The code fits the model-free method known as V-learning (Luckett et, al 2020), and the model assisted weight extension of V-learning and compare the results to the parameter values of the true projection.

The parameters for the true projection are &beta;<sub>0</sub> = -0.012, and &beta;<sub>1</sub> = 0.502.
The code displays the absolute error from these values for the standard and model assisted weight versions of V-learning.

## Use
The code has been simplified as much as possible, but it still requires numpy and scipy to be installed.

The following command will run the example:

```
python runExample.py -n 50 -t 25 -s 1
```

where 'n' is the number of subjects, 't' is the number of observations per subject, and 's' fixes the state of the random number generator (all are integers).
If the command line arguments are not provided, the default is N=50 T=25 and randomSeed=1.

Larger values of 'n' and 't' can take significantly more time to run. I suggest not having 'n>300' or 't>100'. It is also unstable if 'n' or 't' is too small, so 'n>10' and 't>10' is recommended.

At the default values of '-n', '-t' and '-s' the output should look like the following:

```
Simulation with N=50, T=25, and random seed of 1

Absolute error for beta_0:
Standard V-learning        0.1026
Model Assisted V-learning  0.0085

Absolute error for beta_1:
Standard V-learning        0.2406
Model Assisted V-learning  0.0093

```

The default setting is uncharacteristically good for the model assisted V-learning (try changing just the random seed 's'). On average the model assisted method should have smaller error, especially for larger values 'n' and 't'.
