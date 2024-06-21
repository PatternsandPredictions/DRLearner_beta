# Getting started with DRLearner

There are a number of well known agents (RL algorithms) implemented in Acme which may be a good entry point\
to learn how to implement a new agent if one already knows the theory and has experience with other RL frameworks.


## 1. Agent vs Actor disambiguation

The very first diagram from Acme/Overview may confuse one trying to learn the DRLearner implementation.\
Thus below there is an additional info explaining the terminology and how it corresponds to the DRLearner/Acme components.\
https://dm-acme.readthedocs.io/en/latest/user/overview.html

The classic definitions read from Richard S. Sutton and Andrew G. Barto "Reinforcement Learning: An Introduction":

>http://incompleteideas.net/book/ebook/node28.html
>### 3.1 The Agent-Environment Interface
>The reinforcement learning problem is meant to be a straightforward framing of the problem of learning from\
>interaction to achieve a goal. The learner and decision-maker is called the agent. The thing it interacts with,\
>comprising everything outside the agent, is called the environment. These interact continually, the agent\
>selecting actions and the environment responding to those actions and presenting new situations to the agent.\
>The environment also gives rise to rewards, special numerical values that the agent tries to maximize over time.\
>A complete specification of an environment defines a task, one instance of the reinforcement learning problem.

>http://www.incompleteideas.net/book/ebook/node66.html
>### 6.6 Actor-Critic Methods
>Actor-critic methods are TD methods that have a separate memory structure to explicitly represent the policy\
>independent of the value function. The policy structure is known as the actor, because it is used to select\
>actions, and the estimated value function is known as the critic, because it criticizes the actions made by\
>the actor. Learning is always on-policy: the critic must learn about and critique whatever policy is currently\
>being followed by the actor. The critique takes the form of a TD error. This scalar signal is the sole output\
>of the critic and drives all learning in both actor and critic.


## 2. The Essence of Agent57
Transcribed subtitles from the video "Agent57: Outperforming the Atari Human Benchmark" starting time at 00:01:26 up to 00:05:04\
https://slideslive.com/38928122/agent57-outperforming-the-atari-human-benchmark


>*Agent57 adds two main components on top of NGU. The first is a special decomposition of the value function,\
which enables us to learn the intrinsic and extrinsic value functions separately for a family of shared policies.*
>
>*The second is a Meta-controller, which allows us to adapt the intrinsic value function weight beta and\
discount factor gamma over the course of training.*
>
>*The architecture of our agent is based on NGU, which is in turn based on the R2D2 agent.*
>
>*In this setup, we utilize a single GPU learner process which consumes trajectories from a prioritized Replay Buffer.\
The Replay Buffer is fed by a collection of actors, each acting asynchronously and periodically querying the Learner\
for the most up to date weights.*
>
>*In order to learn in hard exploration games, we leverage the Never Give Up.*
>
>*This method produces an intrinsic motivation signal, which combines episodic short term novelty with long term novelty\
via a multiplicative modulation.*
>
>*Random network distillation is used to model the long term novelty, while the episodic novelty is generated from\
a k-nearest neighbor look-up on episodic memory buffer.*
>
>*The embedding stored in this buffer are learned via the inverse dynamics model, which has chosen to incentivize\
the learning representations which capture only those aspects of the state which are controllable.*
>
>*In the original formulation, a family of Q-functions is learned with couple values for beta and gamma, in order to\
mitigate the need to carefully balance the intrinsic and extrinsic rewards scales, both of which will be problem dependent.* 
>
>*Each Actor samples a Q-function uniformly at random at the start of an episode and follows it until termination.*
>
>*At evaluation time only the purely exploitative policy with beta equals zero is used.*
>
>*But it turns out that the use of a single network to represent the combined intrinsic and extrinsic rewards is problematic,\
as the rewards and policies of each in isolation maybe poorly aligned, leading to conflicting gradient updates and\
potentially allowing for one to entirely dominate the learning progress.*
>
>*To mitigate this, we introduce a Q-function parameterization, which utilizes separate networks and backup operators\
to independently learn the intrinsic and extrinsic components of the return, but computing the bootstrap values\
according to the combined greedy policy.*
>
>*For the meta-controller, we choose a simple window UCB Bandit (Upper Confidence Bound), which operates independently\
on each actor as well as the evaluator.*
>
>*Each arm corresponds to a policy induced by a particular beta-gamma pair, and the observed reward of an arm is simply\
the episode return obtained by that policy.*
>
>*The proportion of experience allocated to each policy, and thus vary both per game and over the course of training,\
with virtually no increase in computational costs.*
>
>*These plots show the evaluator armed choice in different games over the course of training, where smaller indices\
correspond to both higher discounts and more exploitative policies.*
>
>*I want to draw particular attention to Skiing. We can see here that initially high variability in the chosen arm\
eventually gives way to only the highest discount arms which are essentials for proper credit assignment in this game.*

Summary

`Agent57  =  NGU  +  intrinsic/extrinsic value function decomposition  +  Meta-Controller adapting intrinsic beta/gamma params`

`Meta-Controller: multiplicative modulation of the return with episodic short-term novelty and long-term novelty`


## 3. Implementation notes: from Agent57 to DRLearner  

>### Curiosity Learning and NGU
>* NGU is a boredom free curiosity-based RL method
>* Curiosity: learning in environments with sparse rewards
>* Give intrinsic rewards to the agent based on its inability to predict actions generating successive states
>* As the agent visits the same (or similar states) it gets better at predicting transitions, thus intrinsic rewards go to 0 (boredom)
>* Another pitfall is rewarding passive observation (agent is rewarded for observing unpredictable noise without acting)
>* Never Give Up solves these problems using concepts of lifelong/episodic curiosity and controllable states

>### NGU Architecture
>* CNN converts screen images to abstract feature representations
>* Embedding features used to learn controllable states and to generate intrinsic rewards
>* Distance between inter-episode controllable states is used to generate intrinsic rewards (episodic curiosity)
>* RND network generates multiplicative constant for the episodic reward (life-long curiosity)
>* Use UVFA to approximate Q(x, a, θ, βi) -> rt = re + βiri
>* Discrete number of β between βmin and βmax where we include 0 and 1
>* Turn off exploratory policy by acting greedily with respect to Q(x, a, θ, 0)
>* Learn to exploit without seeing any extrinsic reward
>* Concatenate one hot encoding of beta to action and both rewards and feed into LSTM core for the agent

>### Meta-Controller
>* Selects which policy to use at training and evaluation time
>* Policies are represented by [β, γ] pair (also called mixtures – larger mixture index means more exploratory behavior)
>* Results in agent learning when it's better to explore and when to exploit
>* Implemented as UCB multi-armed bandit
>* Extrinsic episode returns are used as rewards for the bandit
>* Each actor has its own meta controller
>* Implemented in `drlearner/drlearner/actor_core.py`


### The Lunar Lander learning example
File: `examples/run_lunar_lander.py`

>### Acme
>https://github.com/deepmind/acme
>* A research framework for reinforcement learning
>* Common interface for multiple RL agents
>* Flexible building blocks for most popular RL algorithms
>* Support for two deep-learning back-ends: TF and JAX

Agents built using Acme framework are written with modular acting and learning components. Each Actor component\
has its own associated Environment and Policy copies so it can be trained in the distributed settings.\
Environment should implement dm_env environment interface. The OpenAI Gym and Atari games are common environments\
used for benchmarking RL agents, so they already have the Acme adapters implemented.\
Original Agent57 had a single GPU Learner process consumming trajectories from the piroritized Replay Buffer.

Files:
* `drlearner/core/local_layout.py` - defines a single process agent
* `drlearner/core/distributed_layout.py` - defines a distributed agent
* `drlearner/drlearner/actor.py` - Actor component
* `drlearner/drlearner/actor_core.py`
* `drlearner/drlearner/agent.py` - defines the Agent
* `drlearner/drlearner/builder.py`
* `drlearner/drlearner/config.py` - agent's hyperparams + run configs
* `drlearner/drlearner/distributed_agent.py` - defines the distributed Agent
* `drlearner/drlearner/drlearner_types.py` - DRLearner types definitions
* `drlearner/drlearner/learning.py` - Learner component
* `drlearner/drlearner/lifelong_curiosity.py` - life-long curiosity modulation algorithm
* `drlearner/drlearner/utils.py` - numeric utility functions

>### Hyperparameters
>Exact values can be found in the section G. Hyperparameters of the original parer:
>
>***Agent57: Outperforming the Atari Human Benchmark***\
>https://arxiv.org/pdf/2003.13350.pdf

File: `drlearner/configs/config_lunar_lander.py` - agent's hyperparams + run configs

>### The environment should
>https://github.com/deepmind/dm_env
>* implement the `dm_env.Environment` environment interface
>* define the `dm_env.specs` specifications of its observation, action, and reward spaces 
>* return the `dm_env.TimeStep` object at each timestep
>
>Observations are NumPy Nd-arrays to be processed by the networks 
>* 3D volume representing image for ConvNet
>* 1D vector for MLP networks

### Simplified abstract EnvironmentLoop
```
while True:
  # Make an initial observation.
  step = environment.reset()
  actor.observe_first(step.observation)

  while not step.last():
    # Evaluate the policy and take a step in the environment.
    action = actor.select_action(step.observation)
    step = environment.step(action)

    # Make an observation and update the actor.
    actor.observe(action, next_step=step)
    actor.update()
```

Files: 
* `drlearner/core/environment_loop.py`
* `drlearner/core/loggers/image.py` - saving a video of playing the game
* `drlearner/core/observers/...` - getting states and actions from the game environment? 
* `drlearner/environments/lunar_lander.py` - provides the Lunar Lander environment

>### Networks
>Detailed explanation of networks can be found in section F. Network Architectures of the original paper:
>
>***Agent57: Outperforming the Atari Human Benchmark***\
>https://arxiv.org/pdf/2003.13350.pdf
>
>torso/head terminology is used in Acme when speaking about multi-head neural networks

Files:
* `drlearner/drlearner/networks/networks_zoo/lunar_lander.py` - entry point for Lunar Lander example
* `drlearner/drlearner/networks/networks.py` - wraps 3 following networks into utility dataclass
* `drlearner/drlearner/networks/distilation_network.py` - random network distillation for life-long curiosity 
* `drlearner/drlearner/networks/embedding_network.py` - image to embedding
* `drlearner/drlearner/networks/uvfa_network.py` - universal value function approximator
* `drlearner/drlearner/networks/uvfa_torso.py`
* `drlearner/drlearner/networks/policy_networks.py` - policy function
                        
### Agent's Performance Evaluation
Files:
* `drlearner/drlearner/utils/stats.py` - stats metrics to evaluate the agent's performance
* `drlearner/drlearner/utils/utils.py` - evaluator and logger

### Logging & Monitoring
File: `scripts/update_tb.py`


## The role of the dependency libraries, modules, and components

>### JAX
> Networks are implement with JAX, and its companion libs like chex, flax, optax, rlax, and haiku
>
>`JAX = Autograd + XLA`\
>Autograd – automatic gradient computations for NumPy functions\
>XLA (Accelerated Linear Algebra) - JIT compiler for linear algebra functions used by TF
>* Differentiate
>* Vectorize
>* Parallelize
>* Just-in-time Compilation

>### Launchpad 
>https://github.com/deepmind/launchpad
>* A programming model for distributed ML research
>* Communication between nodes is implemented via remote procedure calls
>* Program definition is separated from the mechanism used to launch the distributed program
>* It allows to run the same code in different setups – multiple threads, processes, machines or cloud

>### Reverb 
>https://github.com/deepmind/reverb
>* Efficient in-memory data storage
>* Primarily designed for ML, especially for the use-case of replay buffer
>* Multiple data structures representations: LIFO, FIFO, priority queue
>* Supports prioritized sampling, priorities update, etc.

Having a thorough understanding of the fundamental concepts defining Reverb is crucial for efficiently and effectively configuring the training environment.

**To prevent deadlock in a synchronous Reverb setup, it's essential to ensure that the rate limiter isn't triggered, which can be achieved by employing the `MinSize(1)` rate limiter.**
(See https://github.com/google-deepmind/acme/issues/207 for additional information).

When designing a customized training setup, it's recommended to integrate an appropriate sampling strategy and table configuration. Additionally, to maintain prolonged efficiency in a distributed environment, utilizing checkpointing and sharding features can be advantageous.

*Keywords: #Tables, #Item selection strategies, #Rate Limiting, #Sharding, #Checkpointing*