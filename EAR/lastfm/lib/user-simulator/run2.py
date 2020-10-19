# BB-8 and R2-D2 are best friends.

import sys
import os
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import pickle
import torch
import argparse

import time
import numpy as np

from config import global_config as cfg
from epi import run_one_episode, update_PN_model, get_reward
from pn import PolicyNetwork
from SAC import SAC_Net, train_sac
import copy
import random
import json

from collections import defaultdict

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('The max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1

def cuda_(var):
    return var.cuda() if torch.cuda.is_available()else var


def main():
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN')
    parser.add_argument('-playby', type=str, dest='playby', help='playby')
    # options include:
    # AO: (Ask Only and recommend by probability)
    # RO: (Recommend Only)
    # policy: (action decided by our policy network)
    # sac: (action decided by our SAC network)
    parser.add_argument('-fmCommand', type=str, dest='fmCommand', help='fmCommand')
    # the command used for FM, check out /EAR/lastfm/FM/
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer')
    # the optimizer for policy network
    parser.add_argument('-actor_lr', type=float, dest='actor_lr', help='actor learning rate')
    # learning rate of Actor network
    parser.add_argument('-critic_lr', type=float, dest='critic_lr', help='critic learning rate')
    # learning rate of the Critic network
    parser.add_argument('-actor_decay', type=float, dest='actor_decay', help='actor weight decay')
    parser.add_argument('-decay', type=float, dest='decay', help="weight decay for FM model")
    # weight decay
    parser.add_argument('-critic_decay', type=float, dest='critic_decay', help='critic weight decay')
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
    parser.add_argument('-use_sac', type=bool, dest='use_sac', help='use_sac')
    # true if the RL module uses SAC

    A = parser.parse_args()

    cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg, purpose=A.purpose, mod=A.mod, mask=A.mask)

    random.seed(1)

    # we random shuffle and split the valid and test set, for Action Stage training and evaluation respectively, to avoid the bias in the dataset.
    all_list = cfg.valid_list + cfg.test_list
    print('The length of all list is: {}'.format(len(all_list)))
    random.shuffle(all_list)
    the_valid_list = all_list[: int(len(all_list) / 2.0)]
    the_test_list = all_list[int(len(all_list) / 2.0):]

    gamma = A.gamma
    FM_model = cfg.FM_model

    if A.eval == 1:
        if A.mod == 'ear':
            fp = '../../data/PN-model-ear/PN-model-ear.txt'
        if A.mod == 'crm':
            fp = '../../data/PN-model-crm/PN-model-crm.txt'
        if A.initeval == 1:
            if A.mod == 'ear':
                fp = '../../data/PN-model-ear/pretrain-model.pt'
            if A.mod == 'crm':
                fp = '../../data/PN-model-crm/pretrain-model.pt'
    else:
        # means training
        if A.mod == 'ear':
            fp = '../../data/PN-model-ear/pretrain-model.pt'
        if A.mod == 'crm':
            fp = '../../data/PN-model-crm/pretrain-model.pt'
    INPUT_DIM = 0
    if A.mod == 'ear':
        INPUT_DIM = 89
    if A.mod == 'crm':
        INPUT_DIM = 33
    print('fp is: {}'.format(fp))

    # Initialie the policy network to either PolicyNetwork or SAC-Net
    if not A.use_sac:
        PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=64, output_dim=34)
        start = time.time()

        try:
            PN_model.load_state_dict(torch.load(fp))
            print('Now Load PN pretrain from {}, takes {} seconds.'.format(fp, time.time() - start))
        except:
            print('Cannot load the model!!!!!!!!!\n fp is: {}'.format(fp))
            if A.playby == 'policy':
                sys.exit()
        
        if A.optim == 'Adam':
            optimizer = torch.optim.Adam(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
        if A.optim == 'SGD':
            optimizer = torch.optim.SGD(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
        if A.optim == 'RMS':
            optimizer = torch.optim.RMSprop(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)

    else:
        PN_model = SAC_Net(input_dim=INPUT_DIM, dim1=64, output_dim=34, actor_lr=A.actor_lr,
         critic_lr=A.critic_lr, discount_rate=gamma, actor_w_decay=A.actor_decay, critic_w_decay=A.critic_decay)

    numpy_list = list()
    rewards_list = list()
    NUMPY_COUNT = 0

    sample_dict = defaultdict(list)
    conversation_length_list = list()
    for epi_count in range(A.startFrom, A.endAt):
        if epi_count % 1 == 0:
            print('-----\nIt has processed {} episodes'.format(epi_count))
        start = time.time()

        u, item = the_valid_list[epi_count]

        # if A.test == 1 or A.eval == 1:
        if A.eval == 1:
            u, item = the_test_list[epi_count]

        if A.purpose == 'fmdata':
            u, item = 0, epi_count

        if A.purpose == 'pretrain':
            u, item = cfg.train_list[epi_count]

        current_FM_model = copy.deepcopy(FM_model)
        param1, param2 = list(), list()
        param3 = list()
        param4 = list()
        i = 0
        for name, param in current_FM_model.named_parameters():
            param4.append(param)
            # print(name, param)
            if i == 0:
                param1.append(param)
            else:
                param2.append(param)
            if i == 2:
                param3.append(param)
            i += 1
        optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)
        optimizer2_fm = torch.optim.SGD(param4, lr=0.001, weight_decay=A.decay)

        user_id = int(u)
        item_id = int(item)

        write_fp = '../../data/interaction-log/{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
            A.mod.lower(), A.code, A.startFrom, A.endAt, A.actor_lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            A.eval, A.initeval,
            A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask)

        choose_pool = cfg.item_dict[str(item_id)]['categories']

        if A.purpose not in ['pretrain', 'fmdata']:
            # this means that: we are not collecting data for pretraining or fm data
            # then we only randomly choose one start attribute to ask!
            choose_pool = [random.choice(choose_pool)]

        for c in choose_pool:
            with open(write_fp, 'a') as f:
                f.write(
                    'Starting new\nuser ID: {}, item ID: {} episode count: {}, feature: {}\n'.format(user_id, item_id, epi_count, cfg.item_dict[str(item_id)]['categories']))
            start_facet = c
            if A.purpose != 'pretrain' and A.playby != 'sac':
                log_prob_list, rewards = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)
            else:
                if A.playby != 'sac':
                    current_np = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)
                    numpy_list += current_np
                
                else:
                    current_np, current_reward = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)
                    rewards_list += current_reward
                    numpy_list += current_np        

            # update PN model
            if A.playby == 'policy' and A.eval != 1:
                update_PN_model(PN_model, log_prob_list, rewards, optimizer)
                print('updated PN model')
                current_length = len(log_prob_list)
                conversation_length_list.append(current_length)
            # end update

            # Update SAC
            if A.purpose != 'pretrain':
                with open(write_fp, 'a') as f:
                    f.write('Big features are: {}\n'.format(choose_pool))
                    if rewards is not None:
                        f.write('reward is: {}\n'.format(rewards.data.numpy().tolist()))
                    f.write('WHOLE PROCESS TAKES: {} SECONDS\n'.format(time.time() - start))

        # Write to pretrain numpy.
        if A.purpose == 'pretrain':
            if A.playby != 'sac':
                if len(numpy_list) > 5000:
                    with open('../../data/pretrain-numpy-data-{}/segment-{}-start-{}-end-{}.pk'.format(
                            A.mod, NUMPY_COUNT, A.startFrom, A.endAt), 'wb') as f:
                        pickle.dump(numpy_list, f)
                        print('Have written 5000 numpy arrays!')
                    NUMPY_COUNT += 1
                    numpy_list = list()
            else:
                # In SAC mode, collect both numpy_list and rewards_list as training data 
                if len(numpy_list) > 5000 or len(rewards_list) > 5000:
                    assert len(rewards_list) == len(numpy_list), "rewards and state-action pairs have different size!"
                    directory = '../../data/pretrain-sac-numpy-data-{}/segment-{}-start-{}-end-{}.pk'.format(
                            A.mod, NUMPY_COUNT, A.startFrom, A.endAt)
                    rewards_directory = '../../data/pretrain-sac-reward-data-{}/segment-{}-start-{}-end-{}.pk'.format(
                            A.mod, NUMPY_COUNT, A.startFrom, A.endAt)
                    with open(directory, 'wb') as f:
                        pickle.dump(numpy_list, f)
                        print('Have written 5000 numpy arrays for SAC!')
                    with open(rewards_directory, 'wb') as f:
                        pickle.dump(rewards_list, f)
                        print('Have written 5000 rewrds for SAC!')
                    NUMPY_COUNT += 1
                    numpy_list = list()
                    rewards_list = list()
                
        # numpy_list is a list of list.
        # e.g. numpy_list[0][0]: int, indicating the action.
        # numpy_list[0][1]: a one-d array of length 89 for EAR, and 33 for CRM.
        # end write

        # Write sample dict:
        if A.purpose == 'fmdata' and A.playby != 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../data/sample-dict/start-{}-end-{}.json'.format(A.startFrom, A.endAt), 'w') as f:
                    json.dump(sample_dict, f, indent=4)
        # end write
        if A.purpose == 'fmdata' and A.playby == 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../data/sample-dict/valid-start-{}-end-{}.json'.format(A.startFrom, A.endAt),
                          'w') as f:
                    json.dump(sample_dict, f, indent=4)

        check_span = 500
        if epi_count % check_span == 0 and epi_count >= 3 * check_span and cfg.eval != 1 and A.purpose != 'pretrain':
            # We use AT (average turn of conversation) as our stopping criterion
            # in training mode, save RL model periodically
            # save model first
            PATH = '../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
                A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
                A.eval, A.initeval,
                A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, epi_count)
            torch.save(PN_model.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

            # a0 = conversation_length_list[epi_count - 4 * check_span: epi_count - 3 * check_span]
            a1 = conversation_length_list[epi_count - 3 * check_span: epi_count - 2 * check_span]
            a2 = conversation_length_list[epi_count - 2 * check_span: epi_count - 1 * check_span]
            a3 = conversation_length_list[epi_count - 1 * check_span: ]
            a1 = np.mean(np.array(a1))
            a2 = np.mean(np.array(a2))
            a3 = np.mean(np.array(a3))

            with open(write_fp, 'a') as f:
                f.write('$$$current turn: {}, a3: {}, a2: {}, a1: {}\n'.format(epi_count, a3, a2, a1))
            print('current turn: {}, a3: {}, a2: {}, a1: {}'.format(epi_count, a3, a2, a1))

            num_interval = int(epi_count / check_span)
            for i in range(num_interval):
                ave = np.mean(np.array(conversation_length_list[i * check_span: (i + 1) * check_span]))
                print('start: {}, end: {}, average: {}'.format(i * check_span, (i + 1) * check_span, ave))
                PATH = '../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
                    A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo,
                    A.trick,
                    A.eval, A.initeval,
                    A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, (i + 1) * check_span)
                print('Model saved at: {}'.format(PATH))

            if a3 > a1 and a3 > a2:
                print('Early stop of RL!')
                exit()


if __name__ == '__main__':
    main()
