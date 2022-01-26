#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from DataObject import DataObject
from random import sample

class VLearningParameters:
    def __init__(self, penVec=np.array([1*10**(-j) for j in range(1,9)]), nSplits=10, nRounds=5):
        '''
        Initalize object for V-learning hyperparameters

        Parameters
        ----------
        penVec : 1d numpy array, optional
            Different penalization values for V-learning. The default is np.array([1*10**(-j) for j in range(1,9)]).
        nSplits : int, optional
            Number of splits for k-fold cross validation. The default is 10.
        nRounds : int, optional
            Number of rounds of k-fold corss validation. The default is 5.

        Returns
        -------
        None.

        '''
        assert type(penVec) is np.ndarray, 'penVec must be a numpy array'
        assert penVec.ndim==1, 'penVec must be a 1-dimensional numpy array'
        self.penVec = penVec
        try:
            self.nSplits = int(nSplits)
        except:
            assert False,  'nSplits must be castable to integer'
        try:
            self.nRounds = int(nRounds)
        except:
            assert False,  'nRounds must be castable to integer'

def laBeta(featuresCurrent, featuresNext, eeWeights, probs, probsGen, reward, discount, nSub, penVal=0.0):
    '''
    Use linear algebra formulation to calculate V-learning parameter 'beta'

    Parameters
    ----------
    featuresCurrent : 2d numpy array
        Features of current states.
    featuresNext : 2d numpy array
        Features of next states.
    eeWeights : 2d numpy array
        Estimating equation weights.
    probs : 1d numpy array
        Probability of selecting action under evaluation policy.
    probsGen : 1d numpy array
        Probability of selecting action under generating policy.
    reward : 1d numpy array
        Array of rewards.
    discount : float
        Discount factor gamma.
    nSub : int
        Number of subjects in the data.
    penVal : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    1d numpy array
        Value of beta.

    '''

    prob_vec = probs/probsGen

    C = eeWeights.T.dot((prob_vec[:,np.newaxis]*(discount*featuresNext - featuresCurrent)))/nSub
    A = eeWeights.T.dot((reward*prob_vec)[:,np.newaxis])/nSub

    try:
        temp = np.linalg.inv(np.matmul(C.T,C)+penVal*np.eye(C.shape[0]))
    except:
        temp = np.linalg.pinv(np.matmul(C.T,C)+penVal*np.eye(C.shape[0]))

    beta = np.matmul(temp,-np.matmul(C.T,A))

    return np.squeeze(beta)

def getIndexList(nSplits, nSub):
    '''
    Get a list of random splits of the data

    Parameters
    ----------
    nSplits : int
        The number of splits "k" for k-fold validation.
    nSub : int
        The number of subjects in the data.

    Returns
    -------
    indexList : 1d numpy array
        Array of ints corresponding to random splits of the data.

    '''
    indexList = np.array(sample((list(range(nSplits))*int(np.ceil(nSub/nSplits)))[:nSub], nSub))
    return indexList

def splitData(dataObject, indexList, eeWeights, index):
    '''
    Split data and estimating equation weights into training and testing data sets.

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    indexList : 1d nupmy array
        random splits of the data.
    eeWeights : 2d numpy array
        estimating equation weights.
    index : int
        Which index to split on.

    Returns
    -------
    trainData : DataObject
        Training data.
    testData : DataObject
        Testing data.
    trainEeWeights : 2d Numpy array
        training estimating equation weights.
    testEeWeights : 2d Numpy array
        testing estimating equation weights.

    '''
    trainData = DataObject(dataObject.stateDim)
    testData = DataObject(dataObject.stateDim)
    trainData.setData(dataObject.data[np.in1d(dataObject.data[:,0], np.where(indexList!=index)),:],len(np.where(indexList!=index)[0]),dataObject.nObs)
    testData.setData(dataObject.data[np.in1d(dataObject.data[:,0], np.where(indexList==index)),:],len(np.where(indexList==index)[0]),dataObject.nObs)
    trainEeWeights = eeWeights[np.in1d(dataObject.data[:,0], np.where(indexList!=index)),:]
    testEeWeights = eeWeights[np.in1d(dataObject.data[:,0], np.where(indexList==index)),:]
    return trainData, testData, trainEeWeights, testEeWeights

def getBetaVLearn(dataObject, eeWeights, policyObject, featureModel, discount, penVal=0.0):
    '''
    Get beta for V-learning 

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    eeWeights : 2d numpy array
        estimating equation weights.
    policyObject : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    penVal : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    beta : 1d numpy array
        Value of beta.

    '''
    states = dataObject.getStates()
    beta = laBeta(featureModel.getFeaturesCurrent(dataObject), featureModel.getFeaturesNext(dataObject), eeWeights, policyObject.getProbs(states, dataObject.getActions()), dataObject.getProbs(), dataObject.getRewards(), discount, dataObject.nSub, penVal)
    return beta

def tdError(beta, dataObject, featureModel, policyEval, discount):
    '''
    Get temporal difference error for the supplied value of beta

    Parameters
    ----------
    beta : 1d numpy array
        Parameter values for V-learning.
    dataObject : DataObject
        Object containing the data.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    policyEval : PolicyObject
        Evaluation policy.
    discount : float
        Discount factor gamma.

    Returns
    -------
    float
        Temporal difference error.

    '''
    valCurrent = featureModel.getFeaturesCurrent(dataObject).dot(beta)
    valNext = featureModel.getFeaturesNext(dataObject).dot(beta)
    probsGen = dataObject.getProbs()
    probs = policyEval.getProbs(dataObject.getStates(),dataObject.getActions())
    reward = dataObject.getRewards()
    return (((probs/probsGen)*(reward + discount*valNext - valCurrent)).sum(axis=0)/dataObject.nSub)

def selectTuningParameter(penVec, dataObject, eeWeights, policyEval, featureModel, discount, nSplits=5, nRounds=5):
    '''
    Find the best value for the penalty term with k-fold cross validation

    Parameters
    ----------
    penVec : 1d numpy array
            Different penalization values for V-learning.
    dataObject : DataObject
        Object containing the data.
    eeWeights : 2d numpy array
        estimating equation weights.
    policyEval : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    nSplits : int, optional
        The number of splits "k" for k-fold validation.
    nRounds : int, optional
        Number of rounds of k-fold corss validation. The default is 5.

    Returns
    -------
    float
        The optimal penalty value to minimize k-fold TD error.

    '''
    lambdaResFull = np.zeros_like(penVec)
    for k in range(nRounds):
        indexList = getIndexList(nSplits,dataObject.nSub)
        lambdaRes = np.zeros_like(penVec)
        for i in range(nSplits):
            trainData, testData, trainEeWeights, _ = splitData(dataObject, indexList, eeWeights, i)
            for j in range(penVec.size):
                beta = getBetaVLearn(trainData, trainEeWeights, policyEval, featureModel, discount, penVec[j])
                lambdaRes[j] += tdError(beta, testData, featureModel, policyEval, discount)
        lambdaResFull += np.abs(lambdaRes)
    return penVec[np.argmin(np.abs(lambdaResFull))]

def fitVlearning(dataObject, policyEval, featureModel, discount, vParams, eeWeights=None):
    '''
    Find the optimal penalty value and calcualte V-learning solution

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    policyEval : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    vParams : VLearningParameters
        V-learning hyperparameter object.
    eeWeights : 2d numpy array, optional
        Estimating equation weights. The default is None.

    Returns
    -------
    beta : 1d numpy array
        Parameter solution to V-learning approach.
    penVal : float
        The optimal penalty value to minimize k-fold TD error.

    '''
    if eeWeights is None:
        eeWeights = featureModel.getFeaturesCurrent(dataObject)
    penVal = selectTuningParameter(vParams.penVec, dataObject, eeWeights, policyEval, featureModel, discount, vParams.nSplits, vParams.nRounds)
    beta = getBetaVLearn(dataObject, eeWeights, policyEval, featureModel, discount, penVal)
    return beta, penVal




