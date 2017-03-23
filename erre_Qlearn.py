import gym
import numpy
import random
import pandas
import operator
import matplotlib.pyplot as plt

erre_game = 'MountainCar-v0'

# Information about the environment:

env = gym.make(erre_game)
print (env.action_space) # number of possible actions
print (env.observation_space) # number of observations
print (env.observation_space.high)
print (env.observation_space.low)
print (env.observation_space.shape)

###################################
# CLASSES AND FUNCTIONS

# Q Learning class:
class erre_QLearning:

    def __init__(self, policy, alpha, epsilon, gamma):
        self.q_matrix = {} # this is a dictionary
        self.alpha = alpha # Learning rate
        self.epsilon = epsilon
        self.gamma = gamma  # discount
        self.policy = policy # number of possible actio
        
    def __str__(self):
        return str(self.q_matrix) # This is to see the Q matrix (dictionary) 
    
    def obtain_q(self, state, action):
        current_q = self.q_matrix.get((state, action), 0.0)
        return current_q # 0.0 is the default value for each (state, value) pair
    
    def update_q(self, reward, state, action, target_q):
        last_q = self.q_matrix.get((state, action), 0.0)
        if last_q is 0.0:
            self.q_matrix[(state, action)] = reward # This is to initialize 
            # in case it is the first one (first line in the slides, week4)
        else:
        	self.q_matrix[(state, action)] = last_q + self.alpha * (target_q - last_q)
            # This is the update rule :D
            # Con esto actualizamos las q, entenderlo bien, muy importante

###################################
# MAIN ALGORITHM STRUCTURE

if __name__ == '__main__':

    # --> Generate environment and record performance

    env = gym.make(erre_game)

    # --> Q learning algorithm
    erre_Qlearning = erre_QLearning(policy=range(env.action_space.n),
                                    alpha=0.9, 
                                    gamma=0.9, # if this is low, it takes a LOT of time to finish 100 iterations - the higher the better
                                    epsilon=0.1)

    for i_episode in range(5): # number of iterations - change also x_axis!!

        observation = env.reset()

###################################
""" NOTES:
    ######### MODEL PARAMETERS
        # ALPHA:
        higher alpha, near 1, improves the learning rate (this is quicker)
        lower alpha, near 0, is better to "find" solutions
        
        # GAMMA:
        if gamma near 1, you see MANY previous states;
        whereas if gamma is near 0, you only take care of the most recents

        # EPSILON:
        Discount factor? while taking actions, you forget the oldest ones

    ######### ACTIONS
        # 0 = move left
        # 1 = do nothing /random but soft movement
        # 2 = move right
"""
############ UPLOAD MODEL

