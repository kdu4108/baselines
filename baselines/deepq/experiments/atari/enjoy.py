import argparse
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model
from baselines.deepq.experiments.atari.analyze import plotRewardDist


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", default="Amidar", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    # does this clip flag actually work? The agent still receives the actual reward right?
    boolean_flag(parser, "clipped", default=False, help="whether you want to record results in a clipped environment")


    return parser.parse_args()


def make_env(game_name, clip_reward = False):
    """
    Make environment
    Input: game_name, the name of the game (String)
           clip_reward, whether the game should be played with a clipped env
    """
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    # env = wrap_dqn(env)
    env = wrap_dqn(env, clip = clip_reward)

    return env

def play(env, act, stochastic, video_path, clipped, num_trials = 10):
    """
    Have the agent actually play with its learned policy.
    Inputs:
        env, the environment
        act, the agent (?)
        stochastic, boolean for whether to make actions stochastically
        video_path, path to which to save the video of first trial
        clipped, boolean for whether the clipped reward should be recorded
        num_trials, the number of trials to play for
    Outputs:
        Dictionary of nonclipped rewards, clipped rewards, and reward distribution
    """
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    reward = 0
    num_played = 0

    # initialize array for storing rewards for each trial
    rewardArray = []

    # initialize reward frequency dictionary
    rewardFreq = {}

    while num_played < num_trials:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]

        # do action
        obs, rew, done, info = env.step(action)
        if rew > 0:
            print("Rew: " + str(rew))
        # record different types of rewards
        rewardFreq[rew] = rewardFreq.get(rew, 0) + 1

        # clip reward for recording
        # irrelevant because of clipping in wrap_dqn?
        if clipped:
            rew = np.sign(rew)
        
        # add rew to total reward for this trial
        reward += rew

        # reset environment when done
        if done:
            obs = env.reset()

        # only happens when trial is done/that reward is added to info["rewards"]?
        if len(info["rewards"]) > num_episodes:
            print(rewardFreq)
            print(info)

            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            rewardArray.append(reward)
            reward = 0
            num_played += 1
            num_episodes = len(info["rewards"])
    # plotRewardDist(rewardFreq, num_trials)

    # return dictionary of rewards
    return {"Nonclipped": info["rewards"], "Clipped": rewardArray, "RewardDist": rewardFreq}

if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env, clip_reward = args.clipped)
        # env = make_env(args.env)
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        trial_rewards = play(env, act, args.stochastic, args.video, args.clipped)
        print(trial_rewards)

        # with open("results.txt", "a") as text_file:
        #     # text_file.write("Clipped Agent: " + str(trial_rewards) + "\n")        
        #     text_file.write("NonClipped Agent: " + str(trial_rewards) + "\n")        


