#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:28:15 2018

@author: tanzhenyu
"""
import numpy as np
import MDP
import math 
import copy
import random

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions
        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''
        def random_pick(some_list, probabilities):  
          x = random.uniform(0, 1)  
          cumulative_probability = 0.0  
          for item, item_probability in zip(some_list, probabilities):  
                cumulative_probability += item_probability  
                if x < cumulative_probability: break  
          return item 
      
        
        pi_as=np.zeros(self.mdp.nActions)
        state=1
        for i in range(self.mdp.nActions):
            pi_as[i]=math.exp(policyParams[i,state])
        pi_a_s=list(pi_as/sum(pi_as))
        
#        pi_as=np.exp(policyParams[:,state::self.mdp.nStates].reshape(1,self.mdp.nActions))
#        pi_a_s=list(pi_as[0]/sum(pi_as[0]))        
        action=random_pick(range(self.mdp.nActions),pi_a_s)
        
        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited |A|*|S|*|S|
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''
        V_old=np.max(initialR,axis=0)       
        n=np.zeros([self.mdp.nStates,self.mdp.nActions])
        n_1=np.zeros([self.mdp.nStates,self.mdp.nActions,self.mdp.nStates])
        R_old=initialR
        T=defaultT
        
        for i in range(nEpisodes):
            state=s0
            for j in range(nSteps):
                thres=random.random()
                if thres<=epsilon:
                    action=random.sample(range(self.mdp.nActions),1)[0]
                else:         
                    action=np.argmax((R_old+self.mdp.discount*T.dot(V_old)),axis=0)
                [reward,nextState]=self.sampleRewardAndNextState(state,action)        
                n[state,action]+=1
                n_1[state,action,nextstate]+=1
                T[action,state,nextstate]= n_1[state,action,nextstate]/n[state,action]
                R_new[action,state]=(reward+(n[state,action]-1)*R_old[action,state])/n[state,action]
                V_new=np.max((R_new+self.mdp.discount*T.dot(V_old)),axis=0)              
                R_old=copy.deepcopy(R_new)
                V_old=copy.deepcopy(V_new)
                state=copy.deepcopy(nextstate)
        V =V_new
        policy = np.argmax((R_new+self.mdp.discount*T.dot(V_new)),axis=0)

        return [V,policy]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
            
        return policyParams    

