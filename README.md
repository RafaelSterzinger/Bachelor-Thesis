# Weighting Intrinsic and Extrinsic Rewards in Reinforcement Learning
## Abstract
In reinforcement learning, algorithms mainly depend on man-made reward functions which act as feedback to decisions made by a so-called agent.
        In order to learn a desired behaviour, the agent has to optimize these functions via a trial-and-error paradigm.
        This is only feasible if the number of states is low as it has to experience them several times.
        However, states are usually described through images where each combination of pixels depicts a different state and thus, even an 84 x 84 grey-scale image would still allow for endless possibilities.
        For this reason, neural networks were employed which yielded several astonishing results over the last decade.
         
Common for these results is the fact that rewards are awarded densely whereas if rewards are sparse the agent is unable to pick up the desired behaviour at all.
        A prominent example is the game _Montezuma's Revenge_ where the agent's task depends on exploration.
        This issue is tackled by incorporating intrinsic rewards which are based on the naturally occurring phenomenon that babies have an internal urge to discover and acquire new skills, both of which are beneficial to reinforcement learning.

An approach to implement intrinsic motivation is via curiosity which was profoundly explored by [Burda et al. (2018)](https://arxiv.org/abs/1808.04355) where they expressed curiosity as an error reflecting the agent's ability to predict the consequences of its decisions.
They showed that even a purely intrinsically motivated agent is able to obtain extrinsic rewards, and pointed out the limitations of such an approach by presenting the so-called noisy-TV problem where agents get distracted by random white-noise.

Based on these findings, this thesis aims to solve multiple questions concerning the combination of extrinsic and intrinsic rewards.
        Obtained results during the evaluation process suggest that optimally combined rewards do not improve existing approaches, however, they certainly outperformed pure intrinsic ones.
        Additionally, other beneficial properties have been discovered such as a faster and more stable convergence.
        There also seems to be a pattern concerning optimally combined rewards as extrinsic reward signals tended to be weighted more heavily and their heaviness was also correlated with the sparseness of the extrinsic rewards.
        Finally, the impact of the noisy-TV problem had been explored with the posed question on whether or not it is possible to combat its effects via extrinsic motivation.
        It was observed that as soon as the extrinsic reward signal outweighs the intrinsic one the impact of of the noisy-TV is not significant anymore.

### Setup Guide

The required dependencies can be obtained by executing the following command: 
```
pip install -r requirements.txt
```

Afterwards, the training process can be start:
```
python src/run.py
```

Per default, this trains an agent on the game _Breakout_ for 4 x 10^6 frames with equal extrinsic and intrinsic weighting utilizing so-called random features.
The generated data will stored at ``logs/{env}/{env}_{seed}_{feat_learning}_INT-{int_coeff}_EXT-{ext_coeff}``.
Furthermore, the executables ``final_evaluation.sh`` and ``gridearch.sh`` maybe employed to search for an optimal weighting.

Additionally, there is also the option to modify the default settings via different arguments. This project offers great adjustability (see _src/run.py_) and thus, the following table only poses a selection of the most used arguments:

|Long               |Default                     |Type                                 |Description                |
|-------------------|----------------------------|-------------------------------------|---------------------------|
|``--env``          | ``BreakoutNoFrameskip-v4`` | String                              | Environment ID            |
|``--seed``         |  ``0``                     | Integer                             | Seed for RNG              |
|``--feat_learning``| ``none``                   | Choice: none,idf,vaesph,vaenonsph,pix2pix |Type of forward dynamics |
|``--dyn_env``      |``False``                   | Boolean              | Boarder of random noise   |
|``--num_timesteps`` |``1e6``                    | Integer              |Number of training steps   |
|``--ext_coeff``     |``0.5``                    | Float                |Coefficient for extrinsic rewards   |
|``--int_coeff``     |``0.5``                    | Float                |Coefficient for intrinsic rewards   |


With the obtained results, the plots can finally be created. For this purpose, exemplary log files of the game _Breakout_ are already provided:

```
python plots.py
```

This creates the following plot which is identical to _Figure 7.6_ illustrated in this thesis:

![Example Plot of Breakout](https://github.com/RafaelSterzinger/Bachelor-Thesis/blob/master/thesis/figures/breakout/Breakout_eprew_recent.png)

