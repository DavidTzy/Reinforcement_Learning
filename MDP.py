#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:23:33 2018

@author: tanzhenyu
"""

import numpy as np
import copy
class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class
        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''
        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            iterId+=1
            V_new=np.max((self.R+self.discount*self.T.dot(V)),axis=0)
            epsilon=np.linalg.norm(V_new-V,np.inf)
            V=copy.deepcopy(V_new)
            if epsilon<=tolerance:
                break
        return [V,iterId,epsilon] 
    
    
    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''
        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.argmax((self.R+self.discount*self.T.dot(V)),axis=0)
        return policy

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        R_pi=np.zeros(self.nStates)
        T_pi=np.eye(self.nStates)
        for i in range(self.nStates):
            R_pi[i]=self.R[policy[i]][i]
            T_pi[i,]=self.T[policy[i]][i,]
        V = np.dot((np.linalg.inv(np.eye(self.nStates)-self.discount*T_pi)),R_pi)
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy
        iterId = 0
        while iterId<nIterations:
            iterId=iterId+1
            V=self.evaluatePolicy(policy)
            policy_new=self.extractPolicy(V)
            if all(policy_new==policy):
                break
            policy=copy.deepcopy(policy_new)
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV
        iterId = 0
        epsilon = 0       
        R_pi=np.zeros(self.nStates)
        T_pi=np.eye(self.nStates)
        while iterId<nIterations:
            iterId=iterId+1
            for i in range(self.nStates):
                R_pi[i]=self.R[policy[i]][i]
                T_pi[i,]=self.T[policy[i]][i,]
            V_new=R_pi+self.discount*T_pi.dot(V)
            epsilon=np.linalg.norm(V_new-V,np.inf)
            V=copy.deepcopy(V_new)
            if epsilon<=tolerance:
                break           
        return [V,iterId,epsilon]
    
    
    
    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy
        V_old = initialV
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            iterId+=1
            V=self.evaluatePolicyPartially(policy,V_old,nIterations=nEvalIterations)[0]
            policy=self.extractPolicy(V)
            V_new=np.max((self.R+self.discount*self.T.dot(V)),axis=0)           
            epsilon=np.linalg.norm(V_new-V_old,np.inf)
            V_old=V_new            
            if epsilon<=tolerance:
                break
        return [policy,V,iterId,epsilon]
