# Search-based Offline Imitation Learning (SEABO)

Under review, please do not distribute.

## How to run

For IQL+SEABO, on MuJoCo tasks, run
```
CUDA_VISIBLE_DEVICES=0 nohup python main_iql.py --env halfcheetah-medium-expert-v2 --seed 5 > out.log 2>&1 &
```

For IQL+SEABO, on AntMaze tasks, run
```
CUDA_VISIBLE_DEVICES=0 nohup python main_iql.py --env antmaze-medium-diverse-v0 --seed 5 --beta 5.0 --temperature 10.0 --expectile 0.9 > out.log 2>&1 &
```

For IQL+SEABO, on Adroit tasks, run
```
CUDA_VISIBLE_DEVICES=0 nohup python main_iql.py --env pen-human-v0 --seed 1 --no_action_dim --temperature 0.5 --dropout_rate 0.1 > out.log 2>&1 &
```

For TD3_BC+SEABO, on MuJoCo tasks, run
```
CUDA_VISIBLE_DEVICES=0 nohup python main.py --env halfcheetah-medium-expert-v2 --seed 1 --dir logs/halfcheetah-medium-expert-v2/ > out.log 2>&1 &
```

## Key flags

```
python main.py \
       --k 1 \
       # how many neighbors are used
       --beta 0.5 \
       # the weighting coefficient
       --scale 1 \
       # the reward scale
       --mode 'sas' \
       # what to query, contain actions or not
       --no_action_dim \
       # whether to include action dimension in the reward calculation
```