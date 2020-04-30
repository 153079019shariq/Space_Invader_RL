import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import wandb
from wandb.keras import WandbCallback


import numpy as np
import random
import math
import glob
import io
import os
import cv2
import base64
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import tensorflow as tf


import argparse
import os
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind, Monitor
from a2c import Agent
from neural_network import CNN
import imageio
import time
from numpy.random import seed
from tensorflow import set_random_seed



wandb.init(project="supp")
config = wandb.config
save_path = os.path.join('models_entropy_coeff1', "Space_inv_A2C_LSTM_nstep8_MAX_rew_546")
print("Saved_the_model")



def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='SpaceInvaders-v0')
    return parser.parse_args()


def get_agent(env, nsteps=5, nstack=1, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              epsilon=1e-5, alpha=0.99):
    # Note: nstack=1 since frame_stack=True, during training frame_stack=False
    agent = Agent(Network=CNN, ob_space=env.observation_space,
                  ac_space=env.action_space, nenvs=1, nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    return agent







# **** Caution: Do not modify this cell ****
# initialize total reward across episodes
cumulative_reward = 0
episode = 0

def evaluate(episodic_reward, reset=False):
  '''
  Takes in the reward for an episode, calculates the cumulative_avg_reward
    and logs it in wandb. If episode > 100, stops logging scores to wandb.
    Called after playing each episode. See example below.

  Arguments:
    episodic_reward - reward received after playing current episode
  '''
  global episode
  global cumulative_reward
  if reset:
    cumulative_reward = 0
    episode = 0
    
  episode += 1

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if (episode > 100):
    print("Scores from episodes > 100 won't be logged in wandb.")
    return

  wandb.log({'episodic_reward': episodic_reward})
  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode
  wandb.log({'cumulative_avg_reward': cumulative_avg_reward})
  print("Episode {} Episodic_reward {} Cumulative_avg_reward {}".format(episode,episodic_reward,cumulative_avg_reward))
  return cumulative_avg_reward



def main():


  cumulative_avg_rewards = []
  for seed_ in [10, 50, 100, 200, 500]:
    seed(seed_)
    set_random_seed(seed_)
    print("Seed: ",seed_)
    episode = 0
    
    wandb.init(project="qualcomm-evaluation")
    wandb.config.episodes = 100
    # initialize environment
    env_id = get_args().env
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=False)
    env.seed(seed_) 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = get_agent(env)
    agent.load(save_path)
    lstm_state = np.zeros((1,256),dtype=np.float32)
 
    print("Actions available(%d): %r"%(env.action_space.n, env.env.get_action_meanings()))
  
    # run for 100 episodes
    for i in range(wandb.config.episodes):
      # Set reward received in this episode = 0 at the start of the episode
      episodic_reward = 0
      reset = False
  
      #env = gym.wrappers.Monitor(env, 'test/'+str(i), force=True)
 
      obs = env.reset()
      renders = []
      count   = 0
      action_count = 0
      done = False
      done1 = np.array([int(done)])
      while not done:
          obs = np.expand_dims(obs.__array__(), axis=0)
          a, v,lstm_state = agent.step(obs,S_=lstm_state,M_=done1)
          obs, reward, done, info = env.step(a)
          done1 = np.array([int(done)])
          #env.render()
          action_count += 1
          if(done):
            #print(info)
            break
          
          #if(action_count == 50):
          # print("Action_count",action_count)
          # done = True
          # break
          episodic_reward += reward
      
      # call evaluation function - takes in reward received after playing an episode
      # calculates the cumulative_avg_reward over 100 episodes & logs it in wandb
      if(i==0):
        reset = True
  
      cumulative_avg_reward = evaluate(episodic_reward, reset)
  
      # your models will be evaluated on 100-episode average reward
      # therefore, we stop logging after 100 episodes
      if (i >= 99):
        print("*************************************************************")
        print("CUMULATIVE_AVG_REWARD",cumulative_avg_reward)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        cumulative_avg_rewards.append(cumulative_avg_reward)
        tf.reset_default_graph()
        break
  
      env.close() 
  print("Final score: ", np.mean(cumulative_avg_rewards))

if __name__ == '__main__':
    main()


