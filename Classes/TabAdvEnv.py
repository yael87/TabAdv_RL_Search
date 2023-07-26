
import numpy as np
import pandas as pd
import gym as gym
from gym import spaces as spaces
import torch



class TabAdvEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, model, x_adv, y_adv, raw_data_path):
    # init action that we can take
    super(TabAdvEnv, self).__init__()
    self.sample = x_adv
    self.label = int(y_adv['pred'][0])
    self.prob = model.predict_proba(self.sample)[0][0]
    self.target_models = model
    self.n = self.sample.shape[1]
    self.state = np.zeros(2, dtype=np.float32)


    # self.changes = [change_up(), change_down()]

    # define the number of actions that we can take
    # self.action_space = spaces.Discrete(self.n)

    # self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]))  # steer, gas, brake
    # TODO Action space -
    num_features = x_adv.shape[1]  # Replace with the number of features

    # Define the minimum and maximum values for each feature
    # TODO Replace these with the appropriate values for your specific case
    range_features = pd.read_csv(raw_data_path+'/range_features.csv')
    feature_min = []
    feature_max = []
    for i, row in range_features.iterrows():
        if row['const'] == 'p':
            feature_min.append(0)
            feature_max.append(np.inf)
        elif row['const'] == 'n':
            feature_min.append(-np.inf)
            feature_max.append(0)
        elif row['const'] == 'b':
            feature_min.append(0)
            feature_max.append(1)
        else:
            feature_min.append(-np.inf)
            feature_max.append(np.inf)
            
    #feature_min = [0, 1, -1]  # A list of size 120 with minimum values for each feature
    #feature_max = [10, 100, 1]  # A list of size 120 with maximum values for each feature

    # TODO Define which features should be treated as integers
    #self.integer_features = [0, 1, ]  # Replace with the indices of integer features
    integer_features = np.where(range_features['integer'] == 1)[0].tolist()

    # action_space = dict()
    # for i in range(num_features):
    #     if range_features['const'][i] != 'd':
    #         if i in integer_features:
    #             action_space[i] = spaces.Discrete(feature_max[i] - feature_min[i] + 1)
    #         else:
    #             action_space[i] = spaces.Box(low=feature_min[i], high=feature_max[i], shape=(1,), dtype=np.float32)

    #action_space = tuple(action_space)
    self.action_space = spaces.Box(low=np.array(feature_min),
                               high=np.array(feature_max),
                               dtype=np.float32)
    #self.action_space = spaces.Dict(action_space)

    # for now init in actions space, after that expand it
    self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)##represents states
    # observation_spaces = {}
    # for i in range(num_features):
    #     observation_spaces[f'feature_{i}'] = spaces.Box(low=feature_min[i], high=feature_max[i], shape=(1,),
    #                                                     dtype=np.float32)
    #
    # # Define the observation space as a dictionary of Box spaces
    # self.observation_space = spaces.Dict(observation_spaces)

    self.original_sample = x_adv
    self.original_label = y_adv['pred']
    self.original_prob = self.prob.copy()

    self.terminated = False

    self.total_reward = 0

    self.i = 0##
    self.count = 0#how many time I went over the current prompt
    # define for now when to stop apply changes for now, inspired from https://people.cs.pitt.edu/~chang/seke/seke22paper/paper066.pdf
    # eqvilient to shower length in example
    self.number_of_changes = 1000000

    # initialize first observation
    sign = -1 if self.label == 1 else 1
    self.state[0] = self.label
    self.state[1] = np.abs(0.5 - self.prob) * sign

    # self.state = np.array([self.label, np.abs(0.5 - self.prob) * (-1 if self.label == 1 else 1)], dtype=np.float32)

  def step(self, action):
    # update we made a change
    self.count +=1

    label = self.target_models.predict(self.sample)


    change = True if label == self.original_label else False

    if change:
        # Apply the action to modify the features
        modified_sample = np.copy(self.sample)
        for i, action_component in enumerate(action):
            if i in self.integer_features:
                # If the feature is an integer, round the action component to the nearest integer
                modified_sample[i] == round(action_component)
            else:
                # If the feature is a float, apply the action component directly
                modified_sample[i] == action_component


        # self.sample = self.changes[action](self.sample)
        self.sample = modified_sample
        self.label = self.target_models.predict(self.sample)
        self.prob = self.target_models.predict_proba(self.sample)[0]

    #else - change other features
    sign = (-1 if self.label == 1 else 1)
    # self.state[0] = self.label
    # self.state[1] = np.abs(0.5 - self.prob) * sign
    self.state = np.array([self.label, np.abs(0.5 - self.prob) * sign], dtype=np.float32)


    # calculate reward
    L0_dist = torch.linalg.norm(self.original_sample - self.sample, ord=0)
    L2_dist = torch.linalg.norm(self.original_sample - self.sample, ord=2)
    L_inf_dist = torch.linalg.norm(self.original_sample - self.sample, ord=torch.inf)
    
    if self.original_label == 1:
        if self.prob[1] < 0.5:
            reward = 1
        elif self.prob[1] < self.original_prob[1]:
            reward = 0.5
        elif self.prob[1] > self.original_prob[1]:
           reward = -1
    else:
        if self.prob[1] > 0.5:
            reward = 1
        elif self.prob[1] > self.original_prob[1]:
            reward = 0.5
        elif self.prob[1] < self.original_prob[1]:
            reward = -1

    objective = L0_dist/self.n + reward

    # we want to maximzie this, maybe do some hyper parameter for it
    # maybe apply other method for obhjective

    self.number_of_changes -= 1
    if self.label == 1-self.original_label or self.number_of_changes == 0:
        self.terminated = True

    self.reward = objective
    # double chack!!!!!!!!!!!!
    if self.terminated:
        self.reward = self.total_reward

    self.total_reward += self.reward
    observation = self.state
    info = {}

    return np.array(observation).astype(np.float32), self.reward, self.terminated, info


  def reset(self, seed=None, options=None):#init the env ()
        #super().reset(seed=seed, options=options)
        super().reset()
        self.number_of_changes = 1000000
        self.terminated = False
        # reset the environment, all param need to be 0s.
        sign = -1 if self.label == 1 else 1
        self.state[0] = self.label
        self.state[1] = np.abs(0.5 - self.prob) * sign

        self.sample = self.original_sample
        self.label = self.original_label
        self.prob = self.original_prob
        self.total_reward = 0
        return np.array(self.state).astype(np.float32), {}

  def render(self):
    pass

  def close(self):
    pass