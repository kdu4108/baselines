import argparse
import gym
import os
import numpy as np
import tensorflow as tf
import json
from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model
from baselines.deepq.experiments.atari.enjoy import make_env, play

def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", default="Amidar", type=str, help="name of the game")
    # parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    # parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=True, help="whether or not to use dueling model")
    boolean_flag(parser, "clipped", default=False, help="whether you want to record results in a clipped environment")


    return parser.parse_args()

if __name__ == '__main__':
    # modelList = ["shift/model-shift10/model-46000000", 
    # "shift/model-shift100/model-34000000",
    # "shift/model-shift1000/model-40000000",
    # "shift/model-shiftNeg10/model-48386183",
    # "shift/model-shiftNeg100/model-39000000",
    # "shift/model-shiftNeg1000/model-39000000"]

    modelList = ["scale/model-scale100/model-47000000",
    "scale/model-scale10/model-49400276",
    "scale/model-scale1/model-49457737",
    "scale/model-scale_1/model-49468207",
    "scale/model-scale_01/model-49321802",
    "scale/model-scale_001/model-28000000"]
    modelList = list(map(lambda x: "/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardexperiments/" + x, modelList))
    # videoList = ["shift/model-shift10/", 
    # "shift/model-shift100/",
    # "shift/model-shift1000/",
    # "shift/model-shiftNeg10/",
    # "shift/model-shiftNeg100/",
    # "shift/model-shiftNeg1000/"]
    videoList = list(map(lambda x: x[:-14] + "video.mp4", modelList))
    # videoList = list(map(lambda x: "/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardexperiments/" + x + "video.mp4", videoList))

    resultsDict = {}
    args = parse_args()

    # loop through each model and test each model 10 times
    for i in range(0, len(modelList)):
        tf.reset_default_graph()
        with U.make_session(4) as sess:
            env = make_env(args.env, clip_reward = args.clipped)
            act = deepq.build_act(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=dueling_model if args.dueling else model,
                num_actions=env.action_space.n)
            U.load_state(os.path.join(modelList[i], "saved"))
            trial_rewards = play(env, act, args.stochastic, videoList[i], args.clipped, num_trials = 10)
            resultsDict[modelList[i]] = trial_rewards

    with open('rewardScaleResultsEnjoyMany.json', 'w') as outfile:
        json.dump(resultsDict, outfile, indent = 4, sort_keys = True)           