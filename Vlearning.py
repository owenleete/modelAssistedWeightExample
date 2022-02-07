#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from DataObject import DataObject

class VLearningParameters:
    def __init__(self, penaltyVector=np.array([1*10**(-j) for j in range(1,9)]), numCVSplits=10, numCVRounds=5):
        '''
        Initalize object for V-learning hyperparameters

        Parameters
        ----------
        penaltyVector : 1d numpy array, optional
            Different penalization values for V-learning. The default is [0.1, 0.001, ... , 1e-8].
        numCVSplits : int, optional
            Number of splits for k-fold cross validation. The default is 10.
        numCVRounds : int, optional
            Number of rounds of k-fold corss validation. The default is 5.

        Returns
        -------
        None.

        '''
        assert type(penaltyVector) is np.ndarray, 'penaltyVector must be a numpy array'
        assert penaltyVector.ndim==1, 'penaltyVector must be a 1-dimensional numpy array'
        self.penaltyVector = penaltyVector
        try:
            self.numCVSplits = int(numCVSplits)
        except:
            assert False,  'numCVSplits must be castable to integer'
        try:
            self.numCVRounds = int(numCVRounds)
        except:
            assert False,  'numCVRounds must be castable to integer'

def laBeta(featuresCurrent, featuresNext, estimatingEquationWeights, probs, probsGen, reward, discount, nSub, penaltyValue=0.0):
    '''
    Use linear algebra formulation to calculate V-learning parameter 'beta'

    Parameters
    ----------
    featuresCurrent : 2d numpy array
        Features of current states.
    featuresNext : 2d numpy array
        Features of next states.
    estimatingEquationWeights : 2d numpy array
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
    penaltyValue : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    1d numpy array
        Value of beta.

    '''

    prob_vec = probs/probsGen

    C = estimatingEquationWeights.T.dot((prob_vec[:,np.newaxis]*(discount*featuresNext - featuresCurrent)))/nSub
    A = estimatingEquationWeights.T.dot((reward*prob_vec)[:,np.newaxis])/nSub

    try:
        temp = np.linalg.inv(np.matmul(C.T,C)+penaltyValue*np.eye(C.shape[0]))
    except:
        temp = np.linalg.pinv(np.matmul(C.T,C)+penaltyValue*np.eye(C.shape[0]))

    beta = np.matmul(temp,-np.matmul(C.T,A))

    return np.squeeze(beta)

def getIndexList(numCVSplits, nSub):
    '''
    Get a list of random splits of the data

    Parameters
    ----------
    numCVSplits : int
        The number of splits "k" for k-fold validation.
    nSub : int
        The number of subjects in the data.

    Returns
    -------
    indexList : 1d numpy array
        Array of ints corresponding to random splits of the data.

    '''
    indexList = np.random.choice((list(range(numCVSplits))*int(np.ceil(nSub/numCVSplits)))[:nSub], nSub,replace=False)
    return indexList

def splitData(dataObject, indexList, estimatingEquationWeights, index):
    '''
    Split data and estimating equation weights into training and testing data sets.

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    indexList : 1d nupmy array
        random splits of the data.
    estimatingEquationWeights : 2d numpy array
        estimating equation weights.
    index : int
        Which index to split on.

    Returns
    -------
    trainData : DataObject
        Training data.
    testData : DataObject
        Testing data.
    trainestimatingEquationWeights : 2d Numpy array
        training estimating equation weights.
    testestimatingEquationWeights : 2d Numpy array
        testing estimating equation weights.

    '''
    trainData = DataObject(dataObject.stateDim)
    testData = DataObject(dataObject.stateDim)
    trainData.setData(dataObject.data[np.in1d(dataObject.data[:,0], np.where(indexList!=index)),:],len(np.where(indexList!=index)[0]),dataObject.nObs)
    testData.setData(dataObject.data[np.in1d(dataObject.data[:,0], np.where(indexList==index)),:],len(np.where(indexList==index)[0]),dataObject.nObs)
    trainestimatingEquationWeights = estimatingEquationWeights[np.in1d(dataObject.data[:,0], np.where(indexList!=index)),:]
    testestimatingEquationWeights = estimatingEquationWeights[np.in1d(dataObject.data[:,0], np.where(indexList==index)),:]
    return trainData, testData, trainestimatingEquationWeights, testestimatingEquationWeights

def getBetaVLearn(dataObject, estimatingEquationWeights, policyObject, featureModel, discount, penaltyValue=0.0):
    '''
    Get beta for V-learning 

    Parameters
    ----------
    dataObject : DataObject
        Object containing the data.
    estimatingEquationWeights : 2d numpy array
        estimating equation weights.
    policyObject : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    penaltyValue : float, optional
        L1 penalty on the parameters. The default is 0.0.

    Returns
    -------
    beta : 1d numpy array
        Value of beta.

    '''
    states = dataObject.getStates()
    beta = laBeta(featureModel.getFeaturesCurrent(dataObject), featureModel.getFeaturesNext(dataObject), estimatingEquationWeights, policyObject.getProbs(states, dataObject.getActions()), dataObject.getProbs(), dataObject.getRewards(), discount, dataObject.nSub, penaltyValue)
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

def selectTuningParameter(penaltyVector, dataObject, estimatingEquationWeights, policyEval, featureModel, discount, numCVSplits=5, numCVRounds=5):
    '''
    Find the best value for the penalty term with k-fold cross validation

    Parameters
    ----------
    penaltyVector : 1d numpy array
            Different penalization values for V-learning.
    dataObject : DataObject
        Object containing the data.
    estimatingEquationWeights : 2d numpy array
        estimating equation weights (denoted as psi in paper).
    policyEval : PolicyObject
        Evaluation policy.
    featureModel : FeatureModel
        Model for the V-learning feature space.
    discount : float
        Discount factor gamma.
    numCVSplits : int, optional
        The number of splits "k" for k-fold validation.
    numCVRounds : int, optional
        Number of rounds of k-fold corss validation. The default is 5.

    Returns
    -------
    float
        The optimal penalty value to minimize k-fold TD error.

    '''
    penaltyResFull = np.zeros_like(penaltyVector)
    for k in range(numCVRounds):
        indexList = getIndexList(numCVSplits,dataObject.nSub)
        penaltyRes = np.zeros_like(penaltyVector)
        for i in range(numCVSplits):
            trainData, testData, trainEstimatingEquationWeights, _ = splitData(dataObject, indexList, estimatingEquationWeights, i)
            for j in range(penaltyVector.size):
                beta = getBetaVLearn(trainData, trainEstimatingEquationWeights, policyEval, featureModel, discount, penaltyVector[j])
                penaltyRes[j] += tdError(beta, testData, featureModel, policyEval, discount)
        penaltyResFull += np.abs(penaltyRes)
    return penaltyVector[np.argmin(np.abs(penaltyResFull))]

def fitVlearning(dataObject, policyEval, featureModel, discount, vLearningParameters, estimatingEquationWeights=None):
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
    vLearningParameters : VLearningParameters
        V-learning hyperparameter object.
    estimatingEquationWeights : 2d numpy array, optional
        Estimating equation weights (denoted as psi in paper). The default is None.

    Returns
    -------
    beta : 1d numpy array
        Parameter solution to V-learning approach.
    penaltyValue : float
        The optimal penalty value to minimize k-fold TD error.

    '''
    if estimatingEquationWeights is None:
        estimatingEquationWeights = featureModel.getFeaturesCurrent(dataObject)
    penaltyValue = selectTuningParameter(vLearningParameters.penaltyVector, dataObject, estimatingEquationWeights, policyEval, featureModel, discount, vLearningParameters.numCVSplits, vLearningParameters.numCVRounds)
    beta = getBetaVLearn(dataObject, estimatingEquationWeights, policyEval, featureModel, discount, penaltyValue)
    return beta, penaltyValue




