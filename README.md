<img src="data/logo.jpg" width=25% align="right" />

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [PPO1](baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](baselines/ppo2) (Optimized for GPU)
- [TRPO](baselines/trpo_mpi)

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
#Helpful Links
https://github.com/openai/baselines/issues/4#issuecomment-305961399
https://stackoverflow.com/questions/44865253/libcublas-so-8-0-error-with-tensorflow
https://github.com/openai/gym/issues/637 (ssh visual forwarding)
https://github.com/openai/gym/issues/366 (ssh visual forwarding)

