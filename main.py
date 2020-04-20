from subproc_vec_env import SubprocVecEnv
from atari_wrappers import make_atari, wrap_deepmind, Monitor

from neural_network import CNN
from a2c import learn

import os

import gym
import argparse
import logging
import mlflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

MODEL_PATH = 'models_nstep_8'
SEED = 0


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='SpaceInvaders-v0')
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(2*1e7))
    parser.add_argument('--nenv', help='No. of environments', type=int, default=24)
    parser.add_argument('--episode_life', help='one_life=one_game', type=int, default=1)
    return parser.parse_args()


def train(env_id, num_timesteps, num_cpu,episode_life):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(SEED + rank)
            gym.logger.setLevel(logging.WARN)
            mlflow.log_param("EpisodicLiefEnv",episode_life)
            env = wrap_deepmind(env,episode_life=episode_life)

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
    train(args.env, args.steps, num_cpu=args.nenv,episode_life=args.episode_life)


if __name__ == "__main__":
    main()
