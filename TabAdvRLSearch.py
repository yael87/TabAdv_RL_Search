# -*- coding: utf-8 -*-
"""TryingStableBaseLines3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JGNlj3ZfodzPojT91vYG7gH7VRHA-bKp

****In order to run the notebook you need to change the runtime type to GPU****

required pip install packages
"""

print( 0)

!pip install tensorflow==2.8.0
!pip install tensorflow==2.12.0
!pip install gym
!pip install keras
!pip install keras-rl2
!pip install Levenshtein
!pip install redbaron
!pip install gymnasium
!pip install swig
!pip3 install box2d-py
!pip install --upgrade ipykernel
!pip install pygame --user
!apt-get update && apt-get install ffmpeg freeglut3-dev xvfb  # For visualization
!pip install Levenshtein
!pip install redbaron
!pip install gymnasium
!pip install "stable-baselines3[extra]>=2.0.0a4"

!pip install --upgrade protoc==3.19.0####tensorboard==2.9.1#"protobuf>=3.20.1"

"""imports"""

from redbaron import RedBaron
import ast
from collections import defaultdict
import re
import uuid
import random
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet
import json
from urllib import request
import requests
#######import keras
# import gym, keep it like this, do not change
import random
# from gym import Env, keep it like this
# from gym.spaces import Discrete, Box, keep it like this do not change
import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces


import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd


class TabAdvEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self):
    # init action that we can take
    super(TabAdvEnv, self).__init__()
    self.sample = adv_x[0]
    self.label = adv_y['pred']
    self.prob = model.predict_proba(self.sample)
    self.target_models = model
    self.n = self.sample.shape[1]

    # insert into it all changes that can be applied
    self.changes = [random_number(min_val,max_val)]

    # define the number of actions that we can take
    self.action_space = spaces.Discrete(self.n)
    # for now init in actions space, after that expand it
    self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)##represents states
    self.original_sample = adv_x[0]
    self.original_label = adv_y['pred']
    self.original_prob = self.prob.copy()

    self.terminated = False

    self.total_reward = 0

    self.i = 0##
    self.count = 0#how many time I went over the current prompt
    # define for now when to stop apply changes for now, inspired from https://people.cs.pitt.edu/~chang/seke/seke22paper/paper066.pdf
    # eqvilient to shower length in example
    self.number_of_changes = np.inf

    # initialize first observation
    sign = -1 if self.label == 1 else 1
    self.state[0] = self.label
    self.state[1] = np.abs(0.5 - self.prob) * sign

  def step(self, action):
    # update we made a change
    self.count +=1

    # snippet to send to the model(copilot)
    ###TODO - send the manipulated req first!-----------
    label = self.target_models.predict(self.sample)

    # choose if to do the change on prompt or on suffix
    change = True if label == self.original_label else False


    if change:
      self.sample = self.changes[action](self.sample)
      self.label = self.target_models.predict(self.sample)
      self.prob = self.target_models.predict_proba(self.sample)
    
    #else - change other features
    sign = -1 if self.label == 1 else 1
    self.state[0] = self.label
    self.state[1] = np.abs(0.5 - self.prob) * sign


    # calculate reward
    L0_dist = np.linalg.norm(self.original_sample - self.sample, ord=0)
    L2_dist = np.linalg.norm(self.original_sample - self.sample, ord=2)
    L_inf_dist = np.linalg.norm(self.original_sample - self.sample, ord=np.inf)
    
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
    super().reset(seed=seed, options=options)
    self.number_of_changes = np.inf
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

# from stable_baselines3.common.env_checker import check_env
env = TabAdvEnv()
# If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env


# do init in this step
# :param n_steps: The number of steps to run for each environment per update
#        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
#       NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
# :param batch_size: Minibatch size

model = PPO("MlpPolicy", env, verbose=1,  n_steps=20, batch_size=10, tensorboard_log="./PPO_a/")
for row in adv_x:
    model.learn(total_timesteps=200, tb_log_name="first_run", progress_bar=True)
    model.learn(total_timesteps=200, tb_log_name="second_run", reset_num_timesteps=False, progress_bar=True)
    model.learn(total_timesteps=200, tb_log_name="third_run", reset_num_timesteps=False, progress_bar=True)

!pip install tensorboard
from tensorboard import notebook

notebook.start("--logdir ./PPO_a/")

!tensorboard --logdir=./PPO_a/

vec_env = make_vec_env(TabAdvEnv, n_envs=1)

obs = vec_env.reset()
n_steps = 20
adv_x = pd.DataFrame()
adv_y = pd.DataFrame()
env.sample = adv_x[0]
env.label = adv_y['pred']
env.target_models = model

print(env.prompt)
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    print(f"Step {step + 1}")
    print(f"Change that has been made: {str(env.changes[action[0]])}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

