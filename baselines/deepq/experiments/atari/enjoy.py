import argparse
import gym
import os
import numpy as np
import time

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
)
from baselines import bench
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.deepq.experiments.atari.model import model, dueling_model

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    #if game_name == 'Breakout':
    #    env = gym.make(game_name + "NoFrameskip-v0")
    #else:
    env = gym.make(game_name + "NoFrameskip-v4")
    env = bench.Monitor(env, None)
# <<<<<<< HEAD
#     env = wrap_deepmind(env, frame_stack=True, scale=True)
# =======
    env = wrap_dqn(env, clip = False)
# >>>>>>> 42cf7bc2d0ccc530514896da65a9e1d26f7870bf
    return env


def play(game_name, env, act, stochastic, video_path):
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    fig = plt.figure()
    label = ['NoOp',
             'Fire',
             'Up',
             'Right',
             'Left',
             'Down',
             'UpRight', 
             'UpLeft',
             'DownRight',
             'DownLeft',
             'UpFire',
             'RightFire',
             'LeftFire',
             'DownFire',
             'UpRightFire',
             'UpLeftFire',
             'DownRightFire',
             'DownLeftFire']
   
    f = open("out.txt", "w")
    tmp = label[3]
    label[3] = label[4]
    label[4] = tmp
    print("SWAPED!")
    record_file = False
    while True:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action, q_values = act(np.array(obs)[None], stochastic=stochastic)
        if record_file:
            f.write("{}".format(action[0]))
            for val in q_values[0]:
                f.write(", {}".format(val))
            f.write("\n")
        ind = range(len(q_values[0])) 
        
        #print(q_values[0])
        tmp = q_values[0][3]
        q_values[0][3] = q_values[0][4]
        q_values[0][4] = tmp
        #print(q_values[0])

        plt.cla()
        plt.bar(ind, q_values[0])
        plt.title(label[action[0]])
        plt.ylim(ymin = min(q_values[0]) - 0.01, ymax = max(q_values[0]) + 0.01)
        #plt.ylim(ymin = 0.0, ymax = 20.0)
        plt.xticks(ind, label, fontsize=5, rotation=30)
        plt.draw()
        plt.pause(0.01)

        if game_name == "Breakout":
            if action > 3:
                action -= 2
        obs, rew, done, info = env.step(action)
        print("rew :", rew)
        #time.sleep(0.1)
        
        #print(info)
        if done:
            f.close()
            obs = env.reset()
            if video_recorder.enabled:
                video_recorder.close()
                video_recorder.enabled = False
        #if len(info["rewards"]) > num_episodes:
        #    if len(info["rewards"]) == 1 and video_recorder.enabled:
        #        # save video of first episode
        #        print("Saved video.")
        #        video_recorder.close()
        #        video_recorder.enabled = False
        #    print(info["rewards"][-1])
        #    num_episodes = len(info["rewards"])


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        n_actions = env.action_space.n
        if args.env == "Breakout":
            n_actions = 6
        print("# of actions : ", n_actions)
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=n_actions)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(args.env, env, act, args.stochastic, args.video)
