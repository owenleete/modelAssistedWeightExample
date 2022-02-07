#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


ACTION_COST = 0.5
NUMBER_OF_ACTIONS = 2
NUMBER_OF_STATES = 3
STATE_COLUMN = 2
ACTION_COLUMN = 3
NEXT_STATE_COLUMN = 5

##############################
## Define helper functions  ##
##############################
# Helper functions are for consistency between differnt MDP settings
# The complex operations for the model assisted weight method interface with
#   various object classes
# Heper functions define the MDP specific behavior while maintaing the same
#   interface for the object classes
# The helper functions should not need to be called by the user


def draw(probs):
    '''
    This function draws from 0,...,len(probs) according to the supplied probabilities

    Parameters
    ----------
    probs : 1d numpy array
        Vector of probabilities .

    Returns
    -------
    float
        Random number between 0 and len(probs).

    '''
    # normalize probabilities to sum to 1
    probs = probs/np.sum(probs)
    # get cumulative probabilities
    cuProbs = np.cumsum(probs)
    # dram random uniform variate
    randUnif = np.random.uniform(0,1,1)
    # loop over cumulative probabilities
    ## return fisrt that is > randUnif
    for i in range(probs.size):
        if randUnif < cuProbs[i]:
            # return as float
            return i*1.0
    # extra return statement in case of numerical precision issues
    return probs.size-1

def _getNextState(state, action, params):
    '''
    Get the next state of the MDP based on the current state and action

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    action : float
        The action to be applied to the MDP. can be either 0.0 or 1.0.
    params : 3d numpy array
        Parameters governing transition dynamics.

    Returns
    -------
    1d numpy array
        The next state of the MDP.

    '''
    # extract appropriate probablilites from params based on action
    if action == 0:
        transProb = params[0,int(state[0]),:]
    else:
        transProb = params[1,int(state[0]),:]
    # draw next state according to probabilities
    nextState = draw(transProb)
    # return as numpy array
    return np.array([nextState])

def _getInitialState(dim):
    '''
    Get the initial state of the MDP

    Parameters
    ----------
    dim : int
        The number of unique states in state space.

    Returns
    -------
    1d numpy array
        The initial state of the MDP.

    '''
    # draw randomly with equal probability
    return  np.array([draw(np.array([1]*dim)/dim)])


def _getReward(state,action,nextState,rewardVec):
    '''
    Get the real value reward signal for the MDP

    Parameters
    ----------
    state : 1d numpy array
        Current state of the MDP.
    action : float
        The action to be applied to the MDP. can be either 0.0 or 1.0.
    nextState : 1d numpy array
        Next state of the MDP.
    rewardVec : 1d numpy array
        vector containing reward values.

    Returns
    -------
    rewardVal : float
        Reward for given state, action, next state triple.

    '''
    rewardVal = rewardVec[int(nextState[0])] - ACTION_COST*action
    return rewardVal

def _fitTransitionModel(data, dim):
    '''
    Estimate the parameters for the transition dynamics model based on data

    Parameters
    ----------
    data : 2d numpy array
        Matrix of data extracted from DataObject class.
    dim : int
        The number of unique states in state space.

    Returns
    -------
    params : 3d numpy array
        Parameters governing transition dynamics.

    '''
    # initalize parameter array
    ## dimension 1 is action
    ## dimension 2 is current state
    ## dimension 3 is next state
    params = np.zeros((NUMBER_OF_ACTIONS,dim,dim))
    # loop over each data point to count transitions for each state, action, next state combination
    for i in range(data.shape[0]):
        params[int(data[i,ACTION_COLUMN]),int(data[i,STATE_COLUMN]),int(data[i,NEXT_STATE_COLUMN])] += 1
    # normalize across rows to get probabilities
    for i in range(dim):
        params[0,i,:] = params[0,i,:]/np.sum(params[0,i,:])
        params[1,i,:] = params[1,i,:]/np.sum(params[1,i,:])
    return params

######################################
## Define the TransitionModel class ##
######################################

class TransitionModel:
    '''
    A class to define the transition dynamics model for an MDP
    
    Attributes:
       rewardVec : 1d numpy array
           Vector containing reward values
       dim : int
           Number of unique states in state space
       params : 3d numpy array 
           Parameters governing transition dynamics
           
    Methods:   
       __init__(rewardVec,dim)
       setParameters(params)
       getParameters()
       getInitialState()
       fitTransitionModel(data)
       getNextStates(state, action)
       getReward(state, action, nextState)
    
    Notes:
        Method interfaces are standardized.
        Any changes to behaviour needs to be handeled by adding 
        more to __init__ function or with the helper functions.
    '''
    def __init__(self, rewardVec):
        '''
        Initialize object
            
        Parameters
        ----------
        rewardVec : 1d numpy array
            Vector containing reward values for MDP.
        dim : int, optional
             The number of unique states in state space. The default is 3.

        Returns
        -------
        None.
        
        Notes
        -----
        Creates 'params' placeholder for parameters

        '''
        self.dim = NUMBER_OF_STATES
        self.rewardVec = rewardVec
        self.params = np.zeros((NUMBER_OF_ACTIONS,self.dim,self.dim))
    
    def setParameters(self, params):
        '''
        Set the value of the parameters for the transition dynamcs model

        Parameters
        ----------
        params : 3d numpy array
            Parameters governing transition dynamics.

        Returns
        -------
        None.

        '''
        assert params.shape == (NUMBER_OF_ACTIONS,self.dim,self.dim)
        params = params.copy()
        for i in range(self.dim):
            for j in range(NUMBER_OF_ACTIONS):
                params[j,i,:] = params[j,i,:]/np.sum(params[j,i,:])
        self.params = params
    
    def getParameters(self):
        '''
        Returns the parameters for the transition dynamics model

        Returns
        -------
        3d numpy array
            Parameters governing transition dynamics.

        '''
        return self.params
    
    def getInitialState(self):
        '''
        Randomly set the initial state of the MDP

        Returns
        -------
        1d numpy array
            Initial state of the MDP.

        '''
        return _getInitialState(self.dim)
    
    def fitTransitionModel(self, data):
        '''
        Estimate the parameters for the tanstiosn model from data.

        Parameters
        ----------
        data : 2d numpy array
            An array extracted from the DataObject class.

        Returns
        -------
        None.

        '''
        self.params = _fitTransitionModel(data, self.dim)
    
    def getNextState(self, state, action):
        '''
        Update the state of the MPD based on the current state and selected action

        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
        action : float
            The selected action. Can be either 0.0 or 1.0.

        Returns
        -------
        1d numpy array
            Next state of the MDP.

        '''
        return _getNextState(state, action, self.getParameters())
    

    def getReward(self, state, action, nextState):
        '''
        Get the real value reward signal for the MDP

        Parameters
        ----------
        state : 1d numpy array
            Current state of the MDP.
        action : float
            The action to be applied to the MDP. can be either 0.0 or 1.0.
        nextState : 1d numpy array
            Next state of the MDP.
    
        Returns
        -------
        float
            Reward for provided state, action, nextState triple.

        '''
        return _getReward(state,action,nextState, self.rewardVec)
    