# EAR System -- User Manual and Datasets (Beta Version)

This is the Beta version of our User Manual and Datasets.

Feel free to contact Yisong if you have any questions to use our system. (miaoyisong [AT] gmail.com)

This is possibly the first open-source system for conversational recommendation in recent years.

Wish you find it helpful! 😛





Please cite our paper if you use our codes or dataset.

```
@inproceedings{lei2020estimation,
  title={Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems},
  author={Lei, Wenqiang and He, Xiangnan and Miao, Yisong and Wu, Qingyun and Hong, Richang and Kan, Min-Yen and Chua, Tat-Seng},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={304--312},
  year={2020}
}
```

---



**Table of Content:**

[TOC]



## 1. System Overview

**TODO: Yisong: Need to draw an overview figure of our system here.** 

This system is developed in Python and PyTorch.  It is composed of two components: 

- Recommender Component: `lib/FM`

- Conversation Component: `lib/user-simulator`

Note: Due to different settings in Yelp (enumerated questions) and LastFM (binary questions), their codes are stored in different directory with minor differences. However, the command lines below can be used interchangeably.



### 1.1 Dependencies

- Anaconda3 packages (https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
- PyTorch == 1.3.1+cu92

Run following codes to have a quick check if you have all packages ready.

```bash
First clone EAR to your root directory

cd ~/EAR

python package-check.py 
```



## 2. Datasets

We currently put our data on Google Drive, please contact Yisong if you can't download it ( miaoyisong [AT] gmail.com).

Link:

```
https://drive.google.com/open?id=13WcWe9JthbiSjcGeCWSTB6Ir7O6riiYy
```





## 3. Estimation Stage --  Factorization Machine Model

All Functions of our FM model can be realized by changing the parameter in `FM_old_train.py` as shown below:

```python
    parser = argparse.ArgumentParser(description="Run FM")
    parser.add_argument('-lr', type=float, metavar='<lr>', dest='lr', help='lr')
    parser.add_argument('-flr', type=float, metavar='<flr>', dest='flr', help='flr')
    # means the learning rate of feature similarity learning
    parser.add_argument('-reg', type=float, metavar='<reg>', dest='reg', help='reg')
    # regularization
    parser.add_argument('-decay', type=float, metavar='<decay>', dest='decay', help='decay')
    # weight decay
    parser.add_argument('-qonly', type=int, metavar='<qonly>', dest='qonly', help='qonly')
    # means quadratic form only (letting go other terms in FM equation...)
    parser.add_argument('-bs', type=int, metavar='<bs>', dest='bs', help='bs')
    # batch size
    parser.add_argument('-hs', type=int, metavar='<hs>', dest='hs', help='hs')
    # hidden size
    parser.add_argument('-ip', type=float, metavar='<ip>', dest='ip', help='ip')
    # init parameter for hidden
    parser.add_argument('-dr', type=float, metavar='<dr>', dest='dr', help='dr')
    # dropout ratio
    parser.add_argument('-optim', type=str, metavar='<optim>', dest='optim', help='optim')
    # optimizer
    parser.add_argument('-observe', type=int, metavar='<observe>', dest='observe', help='observe')
    # the frequency of doing evaluation
    parser.add_argument('-oldnew', type=str, metavar='<oldnew>', dest='oldnew', help='oldnew')
    # we don't use this parameter now
    parser.add_argument('-pretrain', type=int, metavar='<pretrain>', dest='pretrain', help='pretrain')
    # does it need to load pretrain model?
    parser.add_argument('-uf', type=int, metavar='<uf>', dest='uf', help='uf')
    # update feature
    parser.add_argument('-rd', type=int, metavar='<rd>', dest='rd', help='rd')
    # remove duplicate, we don;t use this parameter now
    parser.add_argument('-useremb', type=int, metavar='<useremb>', dest='useremb', help='user embedding')
    # update user embedding during feature similarity
    parser.add_argument('-freeze', type=int, metavar='<freeze>', dest='freeze', help='freeze')
    # we don't use this param now
    parser.add_argument('-command', type=int, metavar='<command>', dest='command', help='command')
    # command = 6: normal FM
    # command = 8: with our second type of negative sample
    parser.add_argument('-seed', type=int, metavar='<seed>', dest='seed', help='seed')
    # random seed
    A = parser.parse_args()
```



The code to run experiment:



```bash
# Yelp has the slightly better overall performance with Adagrad Optimizer
# LastFM would be better with SGD.
# The differences are minor.

cd ~/EAR/yelp/lib/FM

#FM (basic FM)
CUDA_VISIBLE_DEVICES=0 python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim Ada -observe 1 -oldnew new -pretrain 0 -rd 0 -uf 0 -freeze 0 -command 6 -useremb 1 -seed 330

#FM+A (FM and attribute aware BPR)
CUDA_VISIBLE_DEVICES=1 python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim Ada -observe 1 -oldnew new -pretrain 0 -rd 0 -uf 0 -freeze 0 -command 8 -useremb 1 -seed 330

#FM+A+T (multi-task training for item recommendation and attribute prediction)
python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim Ada -observe 1 -oldnew new -pretrain 2 -rd 0 -uf 1 -freeze 0 -command 8 -useremb 1 -seed 330
-----------------------------------------------

cd ~/EAR/lastfm/lib/FM

#FM
CUDA_VISIBLE_DEVICES=2 python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim SGD -observe 10 -oldnew new -pretrain 0 -rd 0 -uf 0 -freeze 0 -command 6 -useremb 1 -seed 330

#FM + A
CUDA_VISIBLE_DEVICES=2 python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim SGD -observe 10 -oldnew new -pretrain 0 -rd 0 -uf 0 -freeze 0 -command 8 -useremb 1 -seed 330

#FM + A + MT
python FM_old_train.py -lr 0.01 -flr 0.001 -reg 0.002 -decay 0 -qonly 1 -bs 64 -hs 64 -ip 0.01 -dr 0.5 -optim SGD -observe 1 -oldnew new -pretrain 2 -rd 0 -uf 1 -freeze 0 -command 8 -useremb 1 -seed 330
```



## 4. Action Stage & Reflection Stage -- User Simulator

Both Action Stage and Reflection Stage is implemented in the `user-simulator` directory.

This user simulator obeys our Multi-Round Conversational Recommendation Scenario, it can be easily customised to your own usage. 

We have a user-friendly interface, you only need to change the parameter below in `run.py` to do all experiments.

```python
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN')
    parser.add_argument('-playby', type=str, dest='playby', help='playby')
    # options include:
    # AO: (Ask Only and recommend by probability)
    # RO: (Recommend Only)
    # policy: (action decided by our policy network)
    parser.add_argument('-fmCommand', type=str, dest='fmCommand', help='fmCommand')
    # the command used for FM, check out /EAR/lastfm/FM/
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer')
    # the optimizer for policy network
    parser.add_argument('-lr', type=float, dest='lr', help='lr')
    # learning rate of policy network
    parser.add_argument('-decay', type=float, dest='decay', help='decay')
    # weight decay
    parser.add_argument('-TopKTaxo', type=int, dest='TopKTaxo', help='TopKTaxo')
    # how many 2-layer feature will represent a big feature. Only Yelp dataset use this param, lastFM have no effect.
    parser.add_argument('-gamma', type=float, dest='gamma', help='gamma')
    # gamma of training policy network
    parser.add_argument('-trick', type=int, dest='trick', help='trick')
    # whether use normalization in training policy network
    parser.add_argument('-startFrom', type=int, dest='startFrom', help='startFrom')
    # startFrom which user-item interaction pair
    parser.add_argument('-endAt', type=int, dest='endAt', help='endAt')
    # endAt which user-item interaction pair
    parser.add_argument('-strategy', type=str, dest='strategy', help='strategy')
    # strategy to choose question to ask, only have effect
    parser.add_argument('-eval', type=int, dest='eval', help='eval')
    # whether current run is for evaluation
    parser.add_argument('-mini', type=int, dest='mini', help='mini')
    # means `mini`-batch update the FM
    parser.add_argument('-alwaysupdate', type=int, dest='alwaysupdate', help='alwaysupdate')
    # means always mini-batch update the FM, alternative is that only do the update for 1 time in a session.
    # we leave this exploration tof follower of our work.
    parser.add_argument('-initeval', type=int, dest='initeval', help='initeval')
    # whether do the evaluation for the `init`ial version of policy network (directly after pre-train)
    parser.add_argument('-upoptim', type=str, dest='upoptim', help='upoptim')
    # optimizer for reflection stafe
    parser.add_argument('-upcount', type=int, dest='upcount', help='upcount')
    # how many times to do reflection
    parser.add_argument('-upreg', type=float, dest='upreg', help='upreg')
    # regularization term in
    parser.add_argument('-code', type=str, dest='code', help='code')
    # We use it to give each run a unique identifier.
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose')
    # options: pretrain, others
    parser.add_argument('-mod', type=str, dest='mod', help='mod')
    # options: CRM, EAR
    parser.add_argument('-mask', type=int, dest='mask', help='mask')
    # use for ablation study, 1, 2, 3, 4 represent our four segments, {ent, sim, his, len}
```

### 4.1 Action Stage Training and Evaluation

```bash
cd ~/EAR/yelp/lib/user-simulator

# Training
CUDA_VISIBLE_DEVICES=1 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 0 -initeval 0 -trick 0 -mini 0 -alwaysupdate 0 -upcount 0 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear

# Evaluation
CUDA_VISIBLE_DEVICES=2 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0.7 -strategy maxent -startFrom 0 -endAt 20000 -eval 0 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear
```

To see how the systems works:

```bash
cd ~/EAR/statistics/

python conversation-evaluation.py
```

You will see the printed results in this format:

```
total epi: 11891
turn, 1, SR, 0.000
turn, 2, SR, 0.000
turn, 3, SR, 0.000
turn, 4, SR, 0.003
turn, 5, SR, 0.017
turn, 6, SR, 0.043
turn, 7, SR, 0.086
turn, 8, SR, 0.135
turn, 9, SR, 0.180
turn, 10, SR, 0.227
turn, 11, SR, 0.265
turn, 12, SR, 0.304
turn, 13, SR, 0.339
turn, 14, SR, 0.371
turn, 15, SR, 0.398
average: 12.632
```



### 4.2 Ablation Study

- State Vector

```bash
Change the `-mask`  into 1, 2, 3, 4, for ablation study on segments of ent, sim, his, and len respectively

CUDA_VISIBLE_DEVICES=2 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 1 -purpose train -mod ear

CUDA_VISIBLE_DEVICES=0 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 2 -purpose train -mod ear

CUDA_VISIBLE_DEVICES=1 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 3 -purpose train -mod ear

CUDA_VISIBLE_DEVICES=2 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 4 -purpose train -mod ear
```

- Reflection

```bash
We simply need to change 

CUDA_VISIBLE_DEVICES=1 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 0 -alwaysupdate 0 -upcount 0 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear
```



### 4.3 Baselines

All baselines can all be easily implemented in our user simulator. 

- Max Entropy

```
CUDA_VISIBLE_DEVICES=0 python run.py -mt 15 -playby AO -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear
```

- CRM

The training and Evaluation of this model can also be done following a simple `mod` parameter.

```bash
# Training
CUDA_VISIBLE_DEVICES=1 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 0 -initeval 0 -trick 0 -mini 0 -alwaysupdate 0 -upcount 0 -upreg 0.001 -code stable -mask 0 -purpose train -mod crm

# Evaluation
CUDA_VISIBLE_DEVICES=0 python run.py -mt 15 -playby policy -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 0 -purpose train -mod crm
```

- Abs-Greedy

Abs-Greedy Algorithm can be easily realized through our system. It is equivalent to Recommending Only option plus Update mechanism. 

```
CUDA_VISIBLE_DEVICES=0 python run.py -mt 15 -playby RO -optim SGD -lr 0.001 -fmCommand 8 -upoptim SGD  -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxent -startFrom 0 -endAt 20000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 4 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear
```


### 4.4 SAC-EAR
ALL usages of SAC follows exactly the same as above.

- SAC pretrain
```
python SAC.py -inputdim 89 -hiddendim 64 -outputdim 34 -bs 64 -actor_lr 0.001 -critic_lr 0.001 -actor_decay 0 -critic_decay 0 -discount_rate 0.7 -mod 'ear'
```

- EAR with SAC
```bash
# Training
python run2.py -mt 15 -playby sac -optim SGD -actor_lr 0.001 -critic_lr 0.001 -fmCommand 8 -upoptim SGD -actor_decay 0 -decay 0 -critic_decay 0 -TopKTaxo 3 -gamma 0.7 -strategy maxent -startFrom 0 -endAt 
1000 -eval 0 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear -use_sac True

# Evaluation
python run2.py -mt 15 -playby sac -optim SGD -actor_lr 0.001 -critic_lr 0.001 -fmCommand 8 -upoptim SGD -actor_decay 0 -decay 0 -critic_decay 0 -TopKTaxo 3 -gamma 0.7 -strategy maxent -startFrom 0 -endAt 
1000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code stable -mask 0 -purpose train -mod ear -use_sac True
```

- Run with pretrined SAC-EAR
```
python run2.py -mt 15 -playby sac -optim SGD -actor_lr 0.001 -critic_lr 0.001 -fmCommand 8 -upoptim SGD -actor_decay 0 -decay 0 -critic_decay 0 -TopKTaxo 3 -gamma 0.7 -strategy maxent -startFrom 0 -endAt 
1000 -eval 1 -initeval 1 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code stable -mask 0 -purpose pretrain -mod ear -use_sac True
```

## 5. Licence and Patent

**TODO: Yisong**