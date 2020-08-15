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

Before the training process can be started, the required dependencies must first be installed. 
```
pip install -r requirements.txt
```

Afterwards, the training process can be started which per default trains an agent on the game _Breakout_ for 4 x 10^6 frames with equal extrinsic and intrinsic weighting utilizing so-called random features.
```
python src/run.py
```

This will produce a log file which is stored at "/tmp/{env_name}\_{seed}\_{feat_learning}\_INT-{int\_coeff}\_EXT-{ext\_coeff}", e.g. of the following form:


<table class="tg">
  <tr>
    <td class="tg-cly1">advmean</td>
    <td class="tg-cly1">0.028340142</td>
  </tr>
  <tr>
    <td class="tg-cly1">advstd</td>
    <td class="tg-cly1">0.08429557</td>
  </tr>
  <tr>
    <td class="tg-cly1">best_ext_ret</td>
    <td class="tg-cly1">34</td>
  </tr>
  <tr>
    <td class="tg-cly1">epcount</td>
    <td class="tg-cly1">9.09e+03</td>
  </tr>
  <tr>
    <td class="tg-cly1">eplen</td>
    <td class="tg-cly1">2.04e+03</td>
  </tr>
  <tr>
    <td class="tg-cly1">eprew</td>
    <td class="tg-cly1">33.1</td>
  </tr>
  <tr>
    <td class="tg-cly1">eprew_recent</td>
    <td class="tg-cly1">33.1</td>
  </tr>
  <tr>
    <td class="tg-cly1">ev</td>
    <td class="tg-cly1">0.99</td>
  </tr>
  <tr>
    <td class="tg-cly1">n_updates</td>
    <td class="tg-cly1">1148</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_approxkl</td>
    <td class="tg-cly1">1.5858127e-14</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_aux</td>
    <td class="tg-cly1">0.0</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_clipfrac</td>
    <td class="tg-cly1">0.0</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_dyn_loss</td>
    <td class="tg-cly1">0.0025119937</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_ent</td>
    <td class="tg-cly1">0.1387403</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_feat_var</td>
    <td class="tg-cly1">0.034835648</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_pg</td>
    <td class="tg-cly1">-0.036531404</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_tot</td>
    <td class="tg-cly1">-0.033197492</td>
  </tr>
  <tr>
    <td class="tg-cly1">opt_vf</td>
    <td class="tg-cly1">0.0034726528</td>
  </tr>
  <tr>
    <td class="tg-cly1">rank</td>
    <td class="tg-cly1">0</td>
  </tr>
  <tr>
    <td class="tg-cly1">recent_best_ext_ret</td>
    <td class="tg-cly1">None</td>
  </tr>
  <tr>
    <td class="tg-cly1">retmean</td>
    <td class="tg-cly1">4.8263</td>
  </tr>
  <tr>
    <td class="tg-cly1">retstd</td>
    <td class="tg-cly1">0.8363703</td>
  </tr>
  <tr>
    <td class="tg-cly1">rew_mean</td>
    <td class="tg-cly1">0.01604179</td>
  </tr>
  <tr>
    <td class="tg-cly1">tcount</td>
    <td class="tg-cly1">1.86e+07</td>
  </tr>
  <tr>
    <td class="tg-cly1">total_secs</td>
    <td class="tg-cly1">1.77e+04</td>
  </tr>
  <tr>
    <td class="tg-cly1">tps</td>
    <td class="tg-cly1">1.06e+03</td>
  </tr>
  <tr>
    <td class="tg-cly1">ups</td>
    <td class="tg-cly1">0.0648</td>
  </tr>
  <tr>
    <td class="tg-cly1">vpredmean</td>
    <td class="tg-cly1">4.7979603</td>
  </tr>
  <tr>
    <td class="tg-cly1">vpredstd</td>
    <td class="tg-cly1">0.83044344</td>
  </tr>
</table>


With the obtained results, the plots can finally be created. For this purpose exemplary log files of the game _Breakout_ are already provided which are identical to the ones used to create _Figure 7.6_ in the thesis.

```
python plots.py
``````

