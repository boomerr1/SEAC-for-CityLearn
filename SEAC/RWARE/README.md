<p align="center">
 <img width="350px" src="docs/img/rware.png" align="center" alt="Multi-Robot Warehouse (RWARE)" />
 <p align="center">A multi-agent reinforcement learning environment</p>
</p>

**WARNING**: This is a **fork** meant as an archive of the code used in our publications. Development is still active in the [original repository](https://github.com/semitable/robotic-warehouse). Please redirect any issues, PRs, or questions there.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

<h1>Table of Contents</h1>

- [Environment Description](#environment-description)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Dynamics: Collisions](#dynamics-collisions)
  - [Rewards](#rewards)
- [Environment Parameters](#environment-parameters)
  - [Naming Scheme](#naming-scheme)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Consider Citing](#consider-citing)


# Environment Description

The multi-robot warehouse (RWARE) environment simulates a warehouse with robots moving and delivering requested goods. We based the simulator on real-world applications, in which robots pick-up shelves and deliver them to a workstation. Humans access the content of a shelf, and then robots can return them to empty shelf locations. The image on the right helps visualise how the robots might look like in real-life.

The environment is configurable: it allows for different sizes (difficulty), number of agents, communication capabilities, and reward settings (cooperative/individual). Of course, the parameters used in each experiment must be clearly reported to allow for fair comparisons between algorithms.

## Action Space
In this simulation, robots have the following discrete action space:

A={ Turn Left, Turn Right, Forward, Load/Unload Shelf }

The first three actions allow each robot only to rotate and move forward. Loading/Unloading only works when an agent is beneath a shelf on one of the predesignated locations.

## Observation Space
The observation of an agent is partially observable and consists of a 3x3 (configurable) square centred on the agent. Inside this limited grid, all entities are observable:
- The location, the rotation and whether the agent is carrying a shelf.
- The location and rotation of other robots.
- Shelves and whether they are currently in the request queue.

## Dynamics: Collisions
The dynamics of the environment are also of particular interest. Like a real, 3-dimensional warehouse, the robots can move beneath the shelves. Of course, when the robots are loaded, they must use the corridors, avoiding any standing shelves.

Any collisions are resolved in a way that allows for maximum mobility. When two or more agents attempt to move to the same location, we prioritise the one that also blocks others. Otherwise, the selection is done arbitrarily. The visuals below demonstrate the resolution of various collisions.

 Example 1                 |   Example 2               | Example 3
:-------------------------:|:-------------------------:|:-------------------------:
![](docs/img/collision1.gif)  |  ![](docs/img/collision2.gif)  |  ![](docs/img/collision3.gif)

## Rewards
At each time a set number of shelves R is requested. When a requested shelf is brought to a goal location, another shelf is uniformly sampled and added to the current requests. Agents are rewarded for successfully delivering a requested shelf to a goal location, with a reward of 1. A significant challenge in these environments is for agents to deliver requested shelves but also finding an empty location to return the previously delivered shelf. Having multiple steps between deliveries leads a very sparse reward signal.

# Environment Parameters

The multi-robot warehouse task is parameterised by:

- The size of the warehouse which is preset to either tiny (10x11), small (10x20), medium (16x20), or large (16x29).
- The number of agents N.
- The number of requested shelves R. By default R=N, but easy and hard variations of the environment use R = 2N and R = N/2, respectively.

Note that R directly affects the difficulty of the environment. A small R, especially on a larger grid, dramatically affects the sparsity of the reward and thus exploration: randomly bringing the correct shelf becomes increasingly improbable.

## Naming Scheme

While RWARE allows fine tuning of multiple parameters when using the Warehouse class, it also registers multiple default environments with Gym for simplicity.

The registered names look like `rware-tiny-2ag-v1` and might cryptic in the beginning, but it is not actually complicated. Every name always starts with rware. Next, the map size is appended as -tiny, -small, -medium, or -large. The number of robots in the map is selected as Xag with X being a number larger than one (e.g. -4ag for 4 agents). A difficulty modifier is optionally appended in the form of -easy or -hard, making requested shelves twice or half the number of agents (see section Rewards). Finally -v1 is the version as required by OpenAI Gym. In the time of writing all environments are v1, but we will increase it during changes or bugfixes.

A few examples:
```python
env = gym.make("rware-tiny-2ag-v1")
env = gym.make("rware-small-4ag-v1")
env = gym.make("rware-medium-6ag-hard-v1")
```


Of course, more settings are available, but have to be changed during environment creation. For example:
```python
env = gym.make("rware-tiny-2ag-v1", sensor_range=3, request_queue_size=6)
```

A detailed explanation of all parameters can be found [here](https://github.com/semitable/robotic-warehouse/blob/4307b1fe3afa26de4ca4003fd04ab1319879832a/robotic_warehouse/warehouse.py#L132)

# Installation
Assuming you have Git and Python3 (preferably on a virtual environment: venv or Anaconda) installed, you can download and install it using
```sh
git clone git@github.com:uoe-agents/robotic-warehouse.git
cd robotic-warehouse
pip install -e .
```

# Getting Started

RWARE was designed to be compatible with Open AI's Gym framework.

Creating the environment is done exactly as one would create a Gym environment:

```python
import gym
import robootic_warehouse
env = gym.make("rware-tiny-2ag-v1")
```

The number of agents, the observation space, and the action space are accessed using:
```python
env.n_agents  # 2
env.action_space  # Tuple(Discrete(5), Discrete(5))
env.observation_space  # Tuple(Box(XX,), Box(XX,))
```

The returned spaces are from the Gym library (`gym.spaces`) Each element of the tuple corresponds to an agent, meaning that `len(env.action_space) == env.n_agents` and `len(env.observation_space) == env.n_agents` are always true.

The reset and step functions again are identical to Gym:

```python
obs = env.reset()  # a tuple of observations

actions = env.action_space.sample()  # the action space can be sampled
print(actions)  # (1, 0)
n_obs, reward, done, info = env.step(actions)

print(done)    # [False, False]
print(reward)  # [0.0, 0.0]
```
which leaves as to the only difference with Gym: the rewards and the done flag are lists, and each element corresponds to the respective agent.

Finally, the environment can be rendered for debugging purposes:
```python
env.render()
```
and should be closed before terminating:
```python
env.close()
```


# Consider Citing
If you use this environment, consider citing
1. A comperative evaluation of MARL algorithms that includes this environment
```
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```
2. A method that achieves state-of-the-art performance in the robotic warehouse task
```
@inproceedings{christianos2020shared,
 author = {Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {10707--10717},
 publisher = {Curran Associates, Inc.},
 title = {Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf},
 volume = {33},
 year = {2020}
}

```

