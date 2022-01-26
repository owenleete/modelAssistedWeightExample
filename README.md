# Example of model assisted weights method for reinforcement learning
This repo is intended to provide a publicly accessable simple example of a model assisted weights method for reinforcement learning.
The model assisted weight method is proposed in a research paper currently under review, an updated link will be provide when the paper is publicly available.

## Background
Reinforcement learning methods generally come in two types, model-based and model-free. 

Model-based methods assume that the future states of the MDP can be predicted using a transition dynamics model esitmated from data. Model-based methods work well if the dynamics model is correctly specified, but can have high bias if the model is incorrect.

Model-free methods do not use a model for the transition dynamics, but instead model the state-value as a function of the state. For computational reasons, the state-value model is a linear combination of (possibly non-linear) basis functions of the state variables. These methods work well if the state-value model  is correct, but can perform poorly if the model is incorrect.

Model assisted weight is a method that we propose in the paper mentioned above that is designed to have good performance if either the transition dynamics model, or state-value model (but not necessarily both) is correct. If both models are correct, the model assisted weight method is asymptotically optimal in the sense of Godambe (1985). If the transition dynamics model is incorrect but the state value model is correct, the model assisted weight method is consistent for the correct answer. If the state-value model is incorrect but the transition dynamics model is correct, the model assisted weight method is cocsistent for the projection of the true state-values onto the linear span of the basis functions used in the state-value model.

## Projection example
This setting demonstrates the ability of the model assisted weight method to find the projection. We define an MDP where the state-values could be accurately modeled with quadratic bassis functions, but instead we use linear basis functions for the state-value model. The code fits the model-free method known as V-learning (Luckett et, al 2020), and the model assited weight extension of V-learning and compare the results to the parameter values of the true projection.

The parameters for the true projection are &beta;<sub>0</sub> = -0.012, and &beta;<sub>1</sub> = 0.502.
The code displays the absolute error from these values for the standard and model assisted weight versions of V-learning.

## Use
The code has been simplified as much as possible, but it still requires numpy and scipy to be installed.

The following command will run the example:

```
python runExample.py <N> <T> <randomSeed>
```

where 'N' is the number of subjects, 'T' is the number of observations per subject, and 'randomSeed' fixes the state of the random number generator (all are integers).
If the command line argumnets are not provided, the default is N=T=100 and randomSeed=1.
