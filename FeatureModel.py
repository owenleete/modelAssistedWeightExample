#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

FEATURE_DIMENSION = 2


def _getFeature(state):
    '''
    Get the feature space for a single state

    Parameters
    ----------
    state : 1d numpy array
        The state of the MDP.

    Returns
    -------
    features : 1d numpy array
        The feature space of the supplied state.

    '''
    features = np.ones(FEATURE_DIMENSION)
    features[1] = state[0]-1
    return features

def _getFeatures(states):
    '''
    Get the feature space for multiple states

    Parameters
    ----------
    states : 2d numpy array
        Several states of the MDP.

    Returns
    -------
    features : 2d numpy array
        The feature space expansion of the supplied states.

    '''
    n = states.shape[0]
    features = np.zeros((n,FEATURE_DIMENSION))
    for i in range(n):
        features[i,:] = _getFeature(states[i,:])
    return features

class FeatureModel:
    def __init__(self):
        '''
        Initialize feature space object

        Parameters
        ----------
        None

        Returns
        -------
        None.

        '''
        
    def getFeaturesCurrent(self, data):
        '''
        Get the features for the current states in the data

        Parameters
        ----------
        data : DataObject
            An object of the class type DataObject.

        Returns
        -------
        2d numpy array
            The feature space expansion of the current states.

        '''
        return _getFeatures(data.getStates())
    
    def getFeaturesNext(self, data):
        '''
        Get the features for the next states in the data

        Parameters
        ----------
        data : DataObject
            An object of the class type DataObject.

        Returns
        -------
        2d numpy array
            The feature space expansion of the next states.

        '''
        return _getFeatures(data.getNextStates())
    
    def getFeatures(self, states):
        '''
        Get the feature space for multiple states
    
        Parameters
        ----------
        states : 2d numpy array
            Several states of the MDP.
    
        Returns
        -------
        features : 2d numpy array
            The feature space expansion of the supplied states.
    
        '''
        return _getFeatures(states)
    
    def getFeature(self, state):
        '''
        Get the feature space for a single state
    
        Parameters
        ----------
        state : 1d numpy array
            The state of the MDP.
    
        Returns
        -------
        features : 1d numpy array
            The feature space of the supplied state.
    
        '''
        return _getFeature(state)
