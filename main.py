from subproc_vec_env import SubprocVecEnv
from atari_wrappers import make_atari, wrap_deepmind, Monitor

from neural_network import CNN
from a2c import learn

import os

import gym
import argparse
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

MODEL_PATH = 'models_nstep_8'
SEED = 0


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='SpaceInvaders-v0')
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(80e6))
    parser.add_argument('--nenv', help='No. of environments', type=int, default=24)
    return parser.parse_args()

lis =[10,50,100,200,500,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
def train(env_id, num_timesteps, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(SEED + lis[rank])
            gym.logger.setLevel(logging.WARN)
            env = wrap_deepmind(env)

            # wrap the env one more time for getting total reward
            env = Monitor(env, rank)
            return env
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    print("##########################No_of_enviroments###############",env)
    learn(CNN, env, SEED,nsteps=8, total_timesteps=int(num_timesteps * 1.1))
    env.close()
    pass


def main():
    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.env, args.steps, num_cpu=args.nenv)


if __name__ == "__main__":
    main()
