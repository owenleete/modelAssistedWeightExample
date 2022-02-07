#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse

# Import objects/functions to define MDP, selection policy, and V-learning method
from TransitionModel import TransitionModel
from PolicyObject import PolicyObject
from DataObject import DataObject
from FeatureModel import FeatureModel
from Vlearning import fitVlearning, VLearningParameters
from modelAssisted import fitMA, getProjection

NUMBER_OF_ACTIONS = 2
NUMBER_OF_STATES = 3

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple Model assisted weight example')
    parser.add_argument('-n','--num_subjects', help='Number of simulated subjects', required=False)
    parser.add_argument('-t','--num_observations', help='Number of observations per subjects', required=False)
    parser.add_argument('-s','--seed', help='Random seed for data generation', required=False)
    args = vars(parser.parse_args())    
    
    # Number of subjects
    try:
        N=int(args['num_subjects'])
    except:
        N=50
    # Number of observations per subject
    try:
        T=int(args['num_observations'])
    except:
        T=25
    # Set random seed
    try:
        randomSeed=int(args['seed'])
    except:
        randomSeed=1

    # Number of values needed to uniquely identify the state
    stateDimension = 1
    
    # MDP discount parameter
    discount = 0.9
    
    # Vector of rewards
    rewardVec = np.array([ -0.5, 0.7,  -0.25 ])
    
    # True system dynamics / data generating model
    transGen = TransitionModel(rewardVec)
    # Define parameters for generating model
    tParams = np.zeros((NUMBER_OF_ACTIONS,NUMBER_OF_STATES,NUMBER_OF_STATES))
    tParams[0] = np.array([[0.05,0.475,0.475],[0.05,0.85,0.1],[0.15,0.35,0.5]])
    tParams[1] = np.array([[0.6,0.0,0.4],[0.3,0.25,0.45],[0.15,0.8,0.05]])
    # Set model parameters
    transGen.setParameters(tParams)
    
    # Action selection model used to generate the data
    policyGen = PolicyObject(np.zeros(NUMBER_OF_STATES))
    
    # Proposed action selection model to evaluate
    policyEval = PolicyObject(np.array([ 1., -1.0,  -.25 ]))
    
    # Model for the feature space / basis functions for V-learning model
    featureModel = FeatureModel()
    
    # Set up data object
    data = DataObject(stateDimension,randomSeed)
    # Generate data
    data.generate(N, T, policyGen, transGen)
    
    # Set up transiton model for estimated system dynamics
    transModel = TransitionModel(rewardVec)
    # Estimate system dynamics
    transModel.fitTransitionModel(data.data)
    
    # Set up hyper-parameters for V-learning
    ### Use default values for all
    vLearningParameters = VLearningParameters()
    
    # Fit standard V-learning
    betaV, _ = fitVlearning(data, policyEval, featureModel, discount, vLearningParameters)
    # Fit V-learning with model assisted weights
    betaMA = fitMA(data, policyEval, policyGen, transModel, featureModel, discount, vLearningParameters)
    
    # Get true projection value
    betaProj = getProjection(data, policyEval, transGen, featureModel, discount)
    
    print('Simulation with N=',N,', T=', T, ', and random seed of ', randomSeed ,sep='')
    print('')
    print('Absolute error for beta_0:')
    print('Standard V-learning       ', round(abs(betaProj[0]-betaV[0]),4))
    print('Model Assisted V-learning ', round(abs(betaProj[0]-betaMA[0]),4))
    print('')
    print('Absolute error for beta_1:')
    print('Standard V-learning       ', round(abs(betaProj[1]-betaV[1]),4))
    print('Model Assisted V-learning ', round(abs(betaProj[1]-betaMA[1]),4))
    
