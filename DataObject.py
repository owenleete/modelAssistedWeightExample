#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

STATE_START = 2
NEXT_STATE_START = 4
NON_STATE_COLUMNS = 5


##############################
## Define helper functions  ##
##############################
# Helper functions are for consistency between differnt MDP settings
# The complex operations for the model assisted weight method interface with
#   various object classes
# Heper functions define the MDP specific behavior while maintaing the same
#   interface for the object classes
# The helper functions should not need to be called by the user

def _generate(nSub, nObs, stateDim, policyObject, transitionModel, randomSeed):
    '''
    Generate data using transitionModel and policyObject

    Parameters
    ----------
    nSub : int
        Number of subjects.
    nObs : int
        Number of observations per subject.
    stateDim : int
        Number of variables needed to define the state.
    policyObject : PolicyObject class
        An object containing the action selection policy.
    transitionModel : TransitionModel class
        An object containing the transition dynamics model.
    randomSeed : int
        Random seed for reproducibility.

    Returns
    -------
    data : 2d numpy array
        An array containg the data. The size of the array should be (nSub*nObs) by (2*stateDim + 5)

    Notes
    -----
    The columns of the data are:
        subject #, observation #, <state>, action. probability, <nextState>, reward
        <state> and <nextState> will each occupy stateDim columns

    '''
    np.random.seed(randomSeed)
    # create data array
    data = np.zeros(((nSub*nObs),(stateDim*2+NON_STATE_COLUMNS)))
    # initalize counter to determine current row to assign to
    counter = 0
    nextState = np.zeros(stateDim)
    for i in range(nSub):
        for j in range(nObs):
            # if first observation on a subject get initial state
            if (j==0):
                state = transitionModel.getInitialState()
            # otherwise copy state from nextState on last iteration
            else:
                state = nextState.copy()
            # select action
            action, prob = policyObject.getAction(state)
            # update state
            nextState = transitionModel.getNextState(state, action)
            # calculate reward
            reward = transitionModel.getReward(state, action, nextState)
            # convert state and nextState to list
            statelist = [state[0]]
            nextlist = [nextState[0]]
            for k in range(1,stateDim):
                # add in each dimension of state
                statelist += [state[k]]
                nextlist += [nextState[k]]
            # Assign data to row
            data[counter,:] = [i,j]+statelist+[action, prob]+nextlist+[reward]
            counter+=1
    return data

class DataObject:
    '''
    A class for holding, querying, and generating data
    
    Attributes:
        stateDim : int 
            Number of variables needed to define the state.
        randomSeed : int, optional
            Random seed for reproducibility. The default is 1.
    Methods:
        generate(nSub, nObs, policyObject, transitionModel, randomSeed)
        setData(data)
        getData()
        getStates()
        getNextStates()
        getActions()
        getProbs()
        getRewards()
        
    Notes:
        Method interfaces are standardized.
        Any changes to behaviour needs to be handeled by adding 
        more to __init__ function or with the helper functions.
    '''
    
    def __init__(self, stateDim, randomSeed = 1):
        '''
        Initalize DataObject

        Parameters
        ----------
        stateDim : int
            Number of variables needed to define the state.
        randomSeed : int, optional
            Random seed for reproducibility. The default is 1.

        Returns
        -------
        None.

        '''
        self.stateDim = stateDim
        self.randomSeed = randomSeed
    
    def generate(self, nSub, nObs, policyObject, transitionModel, randomSeed = None):
        '''
        Generate data using transitionModel and policyObject

        nSub : int
            Number of subjects.
        nObs : int
            Number of observations per subject.
        policyObject : PolicyObject class
            An object containing the action selection policy.
        transitionModel : TransitionModel class
            An object containing the transition dynamics model.
        randomSeed : int, optional
            Random seed for reproducibility. The default is 1.

        Returns
        -------
        None.

        '''
        if randomSeed==None:
            randomSeed = self.randomSeed
        self.nSub = nSub
        self.nObs = nObs
        self.data = _generate(self.nSub, self.nObs, self.stateDim, policyObject, transitionModel, randomSeed)
      
    def setData(self, data, nSub, nObs):
        '''
        Set the data. For splitting data for cross validation.

        Parameters
        ----------
        data : 2d numpy array
            An array containg the data. The size of the array should be (nSub*nObs) by (2*stateDim + 5)
        nSub : int
            Number of subjects.
        nObs : int
            Number of observations per subject.

        Returns
        -------
        None.

        '''
        self.nSub = nSub
        self.nObs = nObs
        self.data = data

    def getData(self):
        '''
        Return the data array

        Returns
        -------
        2d numpy array
            The array containg the data.

        '''
        return self.data

    def getStates(self):
        '''
        Return the states from the data

        Returns
        -------
        2d numpy array
            The array of states.

        '''
        return self.data[:,STATE_START:STATE_START+self.stateDim]

    def getNextStates(self):
        '''
        Return the next states from the data

        Returns
        -------
        2d numpy array
            The array of nextStates.

        '''
        return self.data[:,NEXT_STATE_START+self.stateDim:NEXT_STATE_START+2*self.stateDim]

    def getActions(self):
        '''
        Return the actions from the data

        Returns
        -------
        1d numpy array
            An array of the actions.

        '''
        return self.data[:,STATE_START+self.stateDim]

    def getProbs(self):
        '''
        Return the probabilities of the actions from the data

        Returns
        -------
        1d numpy array
            An array of the probabilities.

        '''
        return self.data[:,STATE_START+self.stateDim+1]
    
    def getRewards(self):
        '''
        Return the rewards from the data

        Returns
        -------
        1d numpy array
            An array of the rewards.

        '''
        return self.data[:,NEXT_STATE_START+2*self.stateDim]
    
