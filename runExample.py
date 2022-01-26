#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np


# Import objects/functions to define MDP, selection policy, and V-learning method
from TransitionModel import TransitionModel
from PolicyObject import PolicyObject
from DataObject import DataObject
from FeatureModel import FeatureModel
from Vlearning import fitVlearning, VLearningParameters
from modelAssisted import fitMA, getProjection

if __name__ == '__main__':

    # Number of subjects
    try:
        N=int(sys.argv[1])
    except:
        N=100
    # Number of observations per subject
    try:
        T=int(sys.argv[2])
    except:
        T=100
    # Set random seed
    try:
        randomSeed=int(sys.argv[3])
    except:
        randomSeed=1

    # Number of values needed to uniquely identify the state
    stateDimension = 1
    
    # MDP discount parameter
    discount = 0.9
    
    # Vector of rewards
    rewardVec = np.array([ -0.5, 0.7,  -0.25 ])
    
    # True system dynamics / data generating model
    transGen = TransitionModel(rewardVec,3)
    # Define parameters for generating model
    tParams = np.zeros((2,3,3))
    tParams[0] = np.array([[0.05,0.475,0.475],[0.05,0.85,0.1],[0.15,0.35,0.5]])
    tParams[1] = np.array([[0.6,0.0,0.4],[0.3,0.25,0.45],[0.15,0.8,0.05]])
    # Set model parameters
    transGen.setParameters(tParams)
    
    # Action selection model used to generate the data
    policyGen = PolicyObject(np.zeros(3))
    
    # Proposed action selection model to evaluate
    policyEval = PolicyObject(np.array([ 1., -1.0,  -.25 ]))
    
    # Model for the feature space / basis functions for V-learning model
    featureModel = FeatureModel()
    
    # Set up data object
    data = DataObject(stateDimension,randomSeed)
    # Generate data
    data.generate(N, T, policyGen, transGen)
    
    # Set up transiton model for estimated system dynamics
    transModel = TransitionModel(rewardVec,3)
    # Estimate system dynamics
    transModel.fitTransitionModel(data.data)
    
    # Set up hyper-parameters for V-learning
    ### Use default values for all
    vParams = VLearningParameters()
    
    # Fit standard V-learning
    betaV, _ = fitVlearning(data, policyEval, featureModel, discount, vParams)
    # Fit V-learning with model assisted weights
    betaMA = fitMA(data, policyEval, policyGen, transModel, featureModel, discount, vParams)
    
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
    
