# Weighting Intrinsic and Extrinsic Rewards in Reinforcement Learning
## Abstract
<p style="text-align: justify">
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
</p>      

<!-- 
### Setup Guide
To try own datasets download a training and test split from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC), preferably overlapping 30 days, into `data/`

To install the needed dependencies run ```pip install requirements.txt```

Afterwards you can train your own model by specifying the mode and the trainings data
```
python main.py -m train -d AAPL_train.csv
```

Or you can use existing models for evaluation by specifying the mode, the testing data and the model
```
python main.py -m test -d AAPL_test.csv -n model_18_17_06
```
-->
