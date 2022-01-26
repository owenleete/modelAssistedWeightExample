#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cplex
from scipy.optimize import linprog
from Vlearning import fitVlearning

def setValues(tParams, pParams, rewardVec, discount):
    '''
    Calcualte the state-values [see Puterman (2005) or Sutton & Barto (2020) for definition of value] for each of the states

    Parameters
    ----------
    tParams : 3d numpy array
        Parameters governing transition dynamics..
    pParams : 1d numpy array
        Parameters governing action selection policy.
    rewardVec : 1d numpy array
           Vector containing reward values
    discount : float
        Discount factor gamma.

    Returns
    -------
    1d numpy array
        Array of values for each state.

    '''
    params = np.zeros_like(tParams[0,:,:])
    avgReward = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        lp = min(500,pParams[i])
        prob1 = np.exp(lp)/(1+np.exp(lp))
        prob0 = 1-prob1
        params[i,:] = (prob0)*tParams[0,i,:] + prob1*tParams[1,i,:]
        rewardVec0 = rewardVec
        rewardVec1 = rewardVec - 0.5
        avgReward[i] = np.sum(prob0*rewardVec0*tParams[0,i,:] + prob1*rewardVec1*tParams[1,i,:])    
    Aub=-(np.eye(params.shape[0])-discount*params)
    Bub=-avgReward
    c=[1.0]*params.shape[0]
    res = linprog(c, Aub, Bub, np.zeros_like(Aub), np.zeros_like(Bub), (-np.Inf,np.Inf))
    return res.x

def getProjection(dataObject, policyEval, transModel, featureModel, discount):
    '''
    Calcualte the projection of the state-values onto the feature space

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    transModel : TransitionModel
        Transition dynamics model object for the MDP.
    policyEval : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    beta : 1d numpy array
        Parameter solution to the projection of the valuse on the featuere space.

    '''
    values = setValues(transModel.getParameters(), policyEval.params, transModel.rewardVec, discount)
    states = np.unique(dataObject.data[:,2])
    y = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        y[i] = values[int(states[i])]        
    x = featureModel.getFeatures(states[:,np.newaxis])
    # Calculate projection
    beta = np.linalg.lstsq(x,y,rcond=None)[0]
    return beta

def get_eTDE(dataObject, policyEval, transModel, featureModel, discount):
    '''
    Get expected temporal difference error and feature space for each of the states.

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    policyEval : PolicyObject
        Evaluation policy.
    transModel : TransitionModel
        Transition dynamics model object for the MDP.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    eTDE : 2d numpy array
        Expected TD error for each state.
    features : 2d numpy array
        Featuer space for each state.

    '''
    betaProj = getProjection(dataObject, policyEval, transModel, featureModel, discount)
    states = np.array([[x] for x in np.sort(np.unique(dataObject.getStates()))])
    eTDE = np.zeros(states.size)
    for i in range(states.size):
        state = states[i]
        prob = policyEval.getProb(state,1)
        eTDE[i] = (np.sum((1-prob)*transModel.getParameters()[0,int(state[0]),:] * (transModel.rewardVec + discount*featureModel.getFeatures(states).dot(betaProj) - featureModel.getFeature(state).dot(betaProj))) + \
                 np.sum(prob*transModel.getParameters()[1,int(state[0]),:] * ((transModel.rewardVec - 0.5) + discount*featureModel.getFeatures(states).dot(betaProj) - featureModel.getFeature(state).dot(betaProj))) )
    features = featureModel.getFeatures(states)
    return eTDE, features

def getMAweights(dataObject, policyEval, policyGen, transModel, featureModel, discount):
    '''
    Calculate the model assited weights

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    policyEval : PolicyObject
        Evaluation policy.
    policyGen : PolicyObject
        Generating/behaviour/logging policy.
    transModel : TransitionModel
        Transition dynamics model object for the MDP.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.

    Returns
    -------
    MAWeights : 2d numpy array
        Model assisted estimating equation weights.

    '''
    eTDE, features = get_eTDE(dataObject, policyEval, transModel, featureModel, discount)
    featuresVec = np.zeros((dataObject.data.shape[0],2))
    eTDEVec = np.zeros(dataObject.data.shape[0])
    for i in range(dataObject.data.shape[0]):
        featuresVec[i,:] = features[int(dataObject.data[i,2]),:]
        eTDEVec[i] = eTDE[int(dataObject.data[i,2])]
    probsEval = policyEval.getProbs(dataObject.getStates(),dataObject.getActions())
    probsGen = policyGen.getProbs(dataObject.getStates(),dataObject.getActions())

    # Correction Factor for weights
    corrFact = np.sum(((probsEval/probsGen) * eTDEVec)[:,np.newaxis] * featuresVec , axis=0)/np.sum(((probsEval/probsGen) * eTDEVec**2))

    # Calculate model assisted weights
    MAWeights = featuresVec - eTDEVec[:,np.newaxis]*corrFact[np.newaxis,:]
    
    return MAWeights

def fitMA(dataObject, policyEval, policyGen, transModel, featureModel, discount, vParams):
    '''
    Fit the model assisted weight method.

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    policyEval : PolicyObject
        Evaluation policy.
    policyGen : PolicyObject
        Generating/behaviour/logging policy.
    transModel : TransitionModel
        Transition dynamics model object for the MDP.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    vParams : VLearningParameters
        V-learning hyperparameter object.
    discount : float
        Discount factor gamma.

    Returns
    -------
    betaMA : 1d numpy array
        Parameter solution to model assisted approach.

    '''
    MAWeights = getMAweights(dataObject, policyEval, policyGen, transModel, featureModel, discount)
    betaMA, _ = fitVlearning(dataObject, policyEval, featureModel, discount, vParams, eeWeights=MAWeights)
    return betaMA




