import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

from subproc_vec_env import SubprocVecEnv
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
import time

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


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='SpaceInvaders-v0')
    return parser.parse_args()


def get_agent(env, nsteps=8, nstack=1, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              epsilon=1e-5, alpha=0.99):
    # Note: nstack=1 since frame_stack=True, during training frame_stack=False
    agent = Agent(Network=CNN, ob_space=env.observation_space,
                  ac_space=env.action_space, nenvs=6, nsteps=nsteps, nstack=nstack,  #24
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@LOADED_THE_AGENT_SUCCESFULLY&&&&&&&&&&&&&&&&&&&&&&&&&")
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


  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode

  print("Episode: %d"%(episode),cumulative_avg_reward,episodic_reward)

  return cumulative_avg_reward

SEED = 0

def make_env(seed,rank):
        global env_id
        def _thunk():
            env = make_atari(get_args().env)
            env.seed(seed + rank)
            env = wrap_deepmind(env,frame_stack=True, clip_rewards=False, episode_life=False)
            env = Monitor(env, rank)
            return env
        return _thunk


def update_state(env,obs):
        nh, nw, nc = env.observation_space.shape
        nstack  =4
        nenv    =6  #24
        state = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        #print("Shape_of_state",state.shape)
        #print(nc) 
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        state = np.roll(state, shift=-nc, axis=3) #We shift the existing observation and add the new state at the end
        state[:, :, :, -nc:] = obs
        return state

def main():
  
  
  cumulative_avg_rewards = []
  for seed_ in [500,100]:
    seed(seed_)
    set_random_seed(seed_)
    print("Seed: ",seed_)
    episode = 0
  
    # initialize environment
    env_id = get_args().env
    #env = make_atari(env_id)
    #env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=False)
    #env = Monitor(env)
    
    env = SubprocVecEnv([make_env(seed_,i) for i in range(6)]) #24
    print("CHECK_ENV",env.reset().__array__().shape)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = get_agent(env)

    save_path = os.path.join('models_entropy_coeff1', "Space_inv_A2C_LSTM_nstep8_MAX_avg_rew_641_max_rew_4144")
    agent.load(save_path)
    lstm_state = np.zeros((6,256),dtype=np.float32) #24
    
    print("##############################CHECKING_THE_WORKFLOW**************************************")
    #print("Actions available(%d): %r"%(env.action_space.n, env.env.get_action_meanings()))
  
    # run for 100 episodes
    #for i in range(100):
    counter = 0 
    
    episodic_reward_lis =[]
    for i in range(100):
      # Set reward received in this episode = 0 at the start of the episode
      start_time = time.time()
      episodic_reward = np.zeros((6))    #24
      episodic_reward_m = np.zeros((6))  #24

      reset = False
  
      #env = gym.wrappers.Monitor(env, 'test/'+str(i), force=True)
 
      obs = env.reset()
      renders = []
      count   = 0
      action_count = 0
      done = False
      done1 = np.zeros(6) #24
      done2 = np.zeros(6) #24
      while not done:
          a, v,lstm_state = agent.step(obs,S_=lstm_state,M_=done1)
          obs, reward, done1, info = env.step(a,done1,cond="eval")
          done =done2.all()
          if(done):
            episodic_reward_m1 = episodic_reward_m.max()
            print("index {} episodic_reward {}".format(episodic_reward_m.argmax(),episodic_reward_m1))
            break
          if(done1.any()):
            episodic_reward_m[np.logical_and(done2<=0,done1)] = episodic_reward[np.logical_and(done2<=0,done1)]
            for j in np.nonzero(done1)[0]:
                    episodic_reward[j] = 0
          episodic_reward += reward      
          done2 = np.logical_or(done1,done2)

      end_time =time.time()
      print("Time_difference",end_time-start_time)
      if(i==0):
        print("Reset_called seed",seed_)
        reset = True

      cumulative_avg_reward = evaluate(episodic_reward_m1, reset)
    
    tf.reset_default_graph()
    env.close() 
  
    # your models will be evaluated on 100-episode average reward
    # therefore, we stop logging after 100 episodes
    print("*************************************************************")
    print("CUMULATIVE_AVG_REWARD",cumulative_avg_reward)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    cumulative_avg_rewards.append(cumulative_avg_reward)

  print("Final score: ", np.mean(cumulative_avg_rewards))
if __name__ == '__main__':
    main()
