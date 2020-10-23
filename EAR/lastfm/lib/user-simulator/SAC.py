import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import os
import sys
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck
from sklearn.metrics import classification_report
import argparse
from pn import PolicyNetwork
from pathlib import Path, PureWindowsPath
from time import sleep

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def copy_model(from_model, to_model):
    for to_param, from_param in zip(to_model.parameters(), from_model.parameters()):
            to_param.data.copy_(from_param.data.clone())

class SAC_Net(nn.Module):
    def __init__(self, input_dim, output_dim, dim1, actor_lr, 
                critic_lr, discount_rate, actor_w_decay, critic_w_decay):
        '''
        params: see below in main() for the detailed explanations in parser 
            
        '''
        super(SAC_Net, self).__init__()
        self.actor_network = PolicyNetwork(input_dim, dim1, output_dim)
        # Create 2 critic netowrks for the purpose of debiasing value estimation
        self.critic_network = PolicyNetwork(input_dim, dim1, output_dim)
        self.critic2_network = PolicyNetwork(input_dim, dim1, output_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr, weight_decay=actor_w_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr, weight_decay=critic_w_decay)
        self.critic2_optimizer = torch.optim.Adam(self.critic2_network.parameters(),lr=critic_lr, weight_decay=critic_w_decay)
        # Create 2 target networks to stablelize training
        self.critic1_target = PolicyNetwork(input_dim, dim1, output_dim)
        self.critic2_target = PolicyNetwork(input_dim, dim1, output_dim)
        copy_model(self.critic_network, self.critic1_target)
        copy_model(self.critic2_network, self.critic2_target)
        # Define discount_rate
        self.discount_rate = discount_rate

    def produce_action_info(self, state):
        action_probs = F.softmax(self.actor_network(state), dim=-1)
        greedy_action = torch.argmax(action_probs, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().cpu()
        # When action probs has 0 'probabilities'
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action, action_probs, log_action_probs, greedy_action 

    def calc_acttor_loss(self, batch_states, batch_action):
        criterion = nn.CrossEntropyLoss()
        _, action_probs, log_action_probs, _ = self.produce_action_info(batch_states)
        Vpi_1 = self.critic_network(batch_states)
        Vpi_2 = self.critic2_network(batch_states)
        # Target is set to the minimum of value functions to reduce bias
        min_V = torch.min(Vpi_1, Vpi_2)
        policy_loss = (action_probs * (self.discount_rate * log_action_probs - min_V)).sum(dim=1).mean()
        batch_action = cuda_(torch.from_numpy(batch_action).long())
        policy_loss += 0.5*criterion(action_probs, batch_action)
        log_action_probs = torch.sum(log_action_probs *  action_probs, dim=1)
        return policy_loss, log_action_probs

    def calc_critic_loss(self, batch_states, batch_next_states, 
                        batch_action, batch_rewards):
        batch_action = cuda_(torch.LongTensor(batch_action)).reshape(-1,1)
        with torch.no_grad():
            next_state_action, action_probs, log_action_probs, _ = self.produce_action_info(batch_next_states)
            target1 = self.critic1_target(batch_next_states)
            target2 = self.critic2_target(batch_next_states)
            min_next_target = action_probs * (torch.min(target1, target2) - self.discount_rate * log_action_probs)
            min_next_target = min_next_target.sum(dim=1).unsqueeze(-1)
            next_q = batch_rewards + self.discount_rate * min_next_target
        qf1 = self.critic_network(batch_states).gather(1, batch_action)
        qf2 = self.critic2_network(batch_states).gather(1, batch_action)
        next_q = next_q.max(1)[0].unsqueeze(-1)
        qf1_loss = F.mse_loss(qf1, next_q)
        qf2_loss = F.mse_loss(qf2, next_q)
        return qf1_loss, qf2_loss

def validate(purpose, train_list, train_reward, 
            valid_list, valid_reward, test_list, test_reward, model):
    model.eval()

    if purpose == 1:
        data = train_list
        reward = train_reward
    elif purpose == 2:
        data = valid_list
        reward = valid_reward
    else:
        data = test_list
        reward = test_reward

    data = data
    reward = reward
    bs = 256
    max_iter = int(len(data) / bs)
    start = time.time()
    epoch_loss = 0
    correct = 0

    y_true, y_pred = list(), list()
    for iter_ in range(max_iter):
        left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
        data_batch = data[left: right]
        reward_batch = reward[left: right]

        temp_out = np.array([item[0] for item in data_batch])

        a = [item[1] for item in data_batch]
        s = a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()
        temp_target = torch.from_numpy(temp_out).long()

        reward_batch = np.asarray(train_reward[left:right])
        temp_reward = torch.from_numpy(reward_batch)
        temp_reward = cuda_(temp_reward)

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model.actor_network(temp_in)
        y_true += temp_out.tolist()
        pred_result = pred.data.max(1)[1]
        correct += sum(np.equal(pred_result.cpu().numpy(), temp_out))

        y_pred += pred_result.cpu().numpy().tolist()

    print('Validating purpose {} takes {} seconds, cumulative loss is: {}, accuracy: {}%'.format(purpose, time.time() - start, epoch_loss / max_iter, correct * 100.0 / (max_iter * bs)))
    print(classification_report(y_true, y_pred))
    model.train()




    
def train_sac(bs, train_list, valid_list, test_list,
            model, epoch, model_path, train_reward = None, 
            valid_reward = None, test_reward = None):

    print('-------validating before training {} epoch-------'.format(epoch))
    if epoch > 0:
        validate(2, train_list, train_reward, valid_list, valid_reward, 
        test_list, test_reward, model)

    if epoch == 7:
        PATH = model_path
        torch.save(model.state_dict(), PATH)
        print('Model saved at {}'.format(PATH))
        return

    model.train()

    c = list(zip(train_list, train_reward))
    random.shuffle(c)
    train_list, train_reward = zip(*c)

    epoch_loss = 0
    max_iter = int(len(train_list) / bs)
    start = time.time()

    left, right = 0, min(len(train_list), bs)
    for iter_ in range(1, max_iter):
        # Make sure we have enough data to train SAC
        if len(train_list) <= bs:
            print('Not enough data to train the model!')
            break

        next_left, next_right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)

        data_batch = train_list[left: right]
        next_data_batch = train_list[next_left: next_right]

        # Each item is a action-state pair
        # temp_out is a numpy array of actions
        temp_out = np.array([item[0] for item in data_batch])
        #next_temp_out = np.array([item[0] for item in next_data_batch])

        # a/next_a is a list of to store states/next states
        a = [item[1] for item in data_batch]
        next_a = [item[1] for item in next_data_batch]
        s = a[0].shape[0]
        next_s = next_a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)
        next_b = np.concatenate(next_a).reshape(-1, next_s)

        '''    
        temp_in = torch.from_numpy(b).float()

        temp_target = torch.from_numpy(temp_out).long()

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model(temp_in)
        loss = criterion(pred, temp_target)
        '''

        temp_in = torch.from_numpy(b).float()
        temp_in = cuda_(temp_in)

        temp_in_next = torch.from_numpy(next_b).float()
        temp_in_next = cuda_(temp_in_next)
        
        reward_batch = np.asarray(train_reward[left:right])
        temp_reward = torch.from_numpy(reward_batch)
        temp_reward = cuda_(temp_reward)

        actor_loss, _ = model.calc_acttor_loss(temp_in, temp_out)
        q1_loss, q2_loss = model.calc_critic_loss(temp_in, temp_in_next, temp_out, temp_reward)

        # Train actor
        model.actor_optimizer.zero_grad()
        actor_loss.backward()
        model.actor_optimizer.step()

        if train_reward:
            # Train critic
            model.critic_optimizer.zero_grad()
            q1_loss.backward()
            model.critic_optimizer.step()

            model.critic2_optimizer.zero_grad()
            q2_loss.backward()
            model.critic2_optimizer.step()

        epoch_loss += actor_loss.data
        left, right = next_left, next_right

        # Update target critic network with moving average
        for param, target_param in zip(model.critic_network.parameters(), model.critic1_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1-0.005) * target_param.data)
        for param, target_param in zip(model.critic2_network.parameters(), model.critic2_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1-0.005) * target_param.data)

        if iter_ % 500 == 0:
            print('{} seconds to finished {}% cumulative loss is: {}'.format(time.time() - start, iter_ * 100.0 / max_iter, epoch_loss / iter_))     


def main():
    parser = argparse.ArgumentParser(description="Soft Actor Critic Network")
    parser.add_argument('-inputdim', type=int, dest='inputdim', help='input dimension')
    parser.add_argument('-hiddendim', type=int, dest='hiddendim', help='hidden dimension')
    parser.add_argument('-outputdim', type=int, dest='outputdim', help='output dimension')
    parser.add_argument('-bs', type=int, dest='bs', help='batch size')
    parser.add_argument('-actor_lr', type=float, dest='actor_lr', help='actor learning rate')
    parser.add_argument('-critic_lr', type=float, dest='critic_lr', help='critic learning rate')
    parser.add_argument('-actor_decay', type=float, dest='actor_decay', help='weight decay for actor')
    parser.add_argument('-critic_decay', type=float, dest='critic_decay', help='weight decay for critic')
    parser.add_argument('-discount_rate', type=float, dest='discount_rate', help='discount_rate')
    parser.add_argument('-mod', type=str, dest='mod', help='mod') # ear crm

    A = parser.parse_args()
    print('Arguments loaded!')
    if A.mod == 'ear':
        inputdim = 89
    else:
        inputdim = 33
    PN = SAC_Net(input_dim=inputdim, dim1=A.hiddendim, output_dim=A.outputdim, actor_lr=A.actor_lr, critic_lr=A.critic_lr, 
                    discount_rate=A.discount_rate,actor_w_decay=A.actor_decay, critic_w_decay=A.critic_decay)

    cuda_(PN)
    print('Model on GPU')
    data_list = list()
    reward_list = list()
    
    np_dir = "../../data/pretrain-sac-numpy-data-{}".format(A.mod)
    reward_dir = "../../data/pretrain-sac-reward-data-{}".format(A.mod)
    files = os.listdir(np_dir)
    file_paths = [np_dir + "/" + f for f in files]

    reward_files = os.listdir(reward_dir)
    reward_paths = [reward_dir + "/" + r for r in reward_files] 
    is_windows = sys.platform.startswith('win')
    if is_windows:
        file_paths = [PureWindowsPath(file_path) for file_path in file_paths]
        reward_paths = [PureWindowsPath(reward_path) for reward_path in reward_paths]
    
    # Read data files
    i = 0
    for fp in file_paths:
        with open(fp, 'rb') as f:
            try:
                data_list += pickle.load(f)
                i += 1
            except:
                pass
    print('total files: {}'.format(i))

    # Read reward files
    j = 0
    for rp in reward_paths:
        with open(rp, 'rb') as f:
            try:
                reward_list += pickle.load(f)
                j += 1
            except:
                pass
    print('total reward files: {}'.format(j))

    data_list = data_list[: int(len(data_list) / 1.5)]
    reward_list = reward_list[: int(len(reward_list) / 1.5)]
    print('length of data list is: {}, length of reward list is: {}'.format(len(data_list), len(reward_list)))

    # Shuffle both data_list and reward list
    c = list(zip(data_list, reward_list))
    random.shuffle(c)
    data_list, reward_list = zip(*c)

    train_list, train_reward = data_list[: int(len(data_list) * 0.7)], reward_list[: int(len(reward_list) * 0.7)]
    valid_list, valid_reward = data_list[int(len(data_list) * 0.7): int(len(data_list) * 0.9)], reward_list[int(len(reward_list) * 0.7): int(len(reward_list) * 0.9)]
    test_list, test_reward = data_list[int(len(data_list) * 0.9):], reward_list[int(len(reward_list) * 0.9):] 
    print('train length: {}, valid length: {}, test length: {}'.format(len(train_list), len(valid_list), len(test_list)))
    print('train reward length: {}, valid reward length: {}, test reward length: {}'.format(len(train_reward), len(valid_reward), len(test_reward)))
    sleep(1)  # let you see this

    '''
    if A.optim == 'Ada':
        optimizer = torch.optim.Adagrad(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    criterion = nn.CrossEntropyLoss()
    '''

    for epoch in range(8):
        c = list(zip(train_list, train_reward))
        random.shuffle(c)
        train_list, train_reward = zip(*c)
        model_name = '../../data/PN-model-{}/pretrain-sac-model.pt'.format(A.mod)
        train_sac(A.bs, train_list, valid_list, test_list, PN, epoch, model_name, train_reward, valid_reward, test_reward)


if __name__ == '__main__':
    main()