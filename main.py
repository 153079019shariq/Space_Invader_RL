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
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=1.5*1e7)
    parser.add_argument('--nenv', help='No. of environments', type=int, default=24)
    parser.add_argument('-lr',"--lr", help='learning_rate', type=float, default=7e-4)
    return parser.parse_args()


def train(env_id, num_timesteps, num_cpu,lr):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(SEED + rank)
            gym.logger.setLevel(logging.WARN)
            env = wrap_deepmind(env)

            # wrap the env one more time for getting total reward
            env = Monitor(env, rank)
            return env
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    print("##########################No_of_enviroments###############",env)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@Learning_rate {} %%%%%%%%%%%%%%%".format(lr))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@No_of_timesteps {} %%%%%%%%%%%%%%%".format(num_timesteps))
    learn(CNN, env, SEED,nsteps=8, total_timesteps=int(num_timesteps ),lr=lr)
    env.close()
    pass


def main():
    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.env, args.steps, num_cpu=args.nenv,lr=args.lr)


if __name__ == "__main__":
    main()
