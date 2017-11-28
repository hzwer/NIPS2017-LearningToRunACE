# NIPS2017-LearningToRun

A keras solution for 2nd plase [NIPS RL 2017 challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run/leaderboards?challenge_round_id=12).

More details will be added in a few days

## To Run
### preparation
These instructions expect that opensim-rl conda environment is already setup as described in : https://github.com/stanfordnmbl/osim-rl/

```
$ source activate opensim-rl
```

Other dependencies is needed as follow
* Keras(since old version does not support selu activation)
* TensorFlow
* matplotlib
* numpy
* Pyro4
* parse
* pymsgbox(optional)

### parllelism

This version requires farming, before starting `train.py`, you should first start some farms by running `python farm.py` on each <u>SLAVE</u> machine you own. Then  create a `farmlist.py` in the working directory (on the <u>HOST</u> machine) with the following content:

```
farmlist_base = [('127.0.0.1', 4), ('192.168.1.1', 8)]

# a farm of 4 cores is available on localhost, while a farm of 8 available on another machine.

# expand the list if you have more machines.

# this file will be consumed by the host to find the slaves.
```
try `python farm.py --help` to get more information about how to set the environment

More information can be found in https://github.com/ctmakro/stanford-osrl

Thanks to @ctmakro for providing us with this frame

### test

Test the model for 10 times in parallel and calculate the average score

```
python test.py
```

### Demo 

![Demo](https://github.com/hzwer/NIPS2017-LearningToRun/raw/master/demo/hzwer-NIPS2017-LearningToRun.gif)

