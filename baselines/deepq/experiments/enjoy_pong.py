import gym
# from baselines import deepq
from baselines.deepq.__init__ import wrap_atari_dqn


def main():
    env = gym.make("PongNoFrameskip-v4")
#     env = deepq.wrap_atari_dqn(env)
    env = wrap_atari_dqn(env)
    act = deepq.load("pong_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
