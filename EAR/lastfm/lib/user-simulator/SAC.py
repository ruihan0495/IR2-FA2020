import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck
from pn import PolicyNetwork

'''
Note: we have contacted the authors for the data file ptetrin-numpy-data-ear
to re-train the actor-critc policy network from scratch. We also plan to incorporate the 
online update for the policy network in reflection stage.
'''

def copy_model(from_model, to_model):
    for to_param, from_param in zip(to_model.parameters(), from_model.parameters()):
            to_param.data.copy_(from_param.data.clone())

class SAC_Net(nn.Module):
    def __init__(self, input_dim, output_dim, dim1, alpha, discount_rate):
        '''
        params:
            alpha - the learning rate when train the actor and critic losses 
        '''
        super(SAC_Net, self).__init__()
        self.actor_network = PolicyNetwork(input_dim, dim1, output_dim)
        # Create 2 critic netowrks for the purpose of debiasing value estimation
        self.critic_network = PolicyNetwork(input_dim, dim1, output_dim)
        self.critic2_network = PolicyNetwork(input_dim, dim1, output_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters())
        self.critic2_optimizer = torch.optim.Adam(self.critic2_network.parameters())
        # Create 2 target networks to stablelize training
        self.critic1_target = PolicyNetwork(input_dim, dim1, output_dim)
        self.critic2_target = PolicyNetwork(input_dim, dim1, output_dim)
        copy_model(self.critic_network, self.critic1_target)
        copy_model(self.critic2_network, self.critic2_target)
        # Define learing rate and discount_rate
        self.alpha = alpha
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

    def calc_acttor_loss(self, batch_states):
        action, action_probs, log_action_probs, _ = self.produce_action_info(batch_states)
        Vpi_1 = self.critic_network(batch_states)
        Vpi_2 = self.critic2_network(batch_states)
        min_V = torch.min(Vpi_1, Vpi_2)
        policy_loss = action_probs * (self.alpha * log_action_probs - min_V).sum(dim=1).mean()
        log_action_probs = torch.sum(log_action_probs *  action_probs, dim=1)
        return policy_loss, log_action_probs

    def calc_critic_loss(self, batch_states, batch_next_states, 
                        batch_action, batch_rewards):
        with torch.np_grad():
            next_state_action, action_probs, log_action_probs, _ = self.produce_action_info(batch_next_states)
            target1 = self.critic1_target(batch_next_states)
            target2 = self.critic2_target(batch_next_states)
            min_next_target = action_probs * (torch.min(target1, target2) - self.alpha * log_action_probs)
            min_next_target = min_next_target.sum(dim=1).unsqueeze(-1)
            next_q = batch_rewards + self.discount_rate * min_next_target
        
        qf1 = self.critic_network(batch_states).gather(1, batch_action)
        qf2 = self.critic2_network(batch_states).gather(1, batch_action)
        qf1_loss = F.mse_loss(qf1, next_q)
        qf2_loss = F.mse_loss(qd2, next_q)
        return qf1_loss, qf2_loss

    
def train_sac(bs, train_list, valid_list, test_list,
            model, epoch, model_path, reward_list = None):
    '''
    params: 
        train_list/valid_list/test_list: 

        reward_list:

        action_list:
    '''
    print('-------validating before training {} epoch-------'.format(epoch))
    if epoch > 0:
        validate(2, train_list, valid_list, test_list, model)

    if epoch == 7:
        PATH = model_path
        torch.save(model.state_dict(), PATH)
        print('Model saved at {}'.format(PATH))
        return

    model.train()
    random.shuffle(train_list)
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

        # temp_reward should get from the run_one_episode

        actor_loss, _ = model.calc_acttor_loss(temp_in)
        q1_loss, q2_loss = model.calc_critic_loss(temp_in, temp_in_next, temp_out, temp_reward)

        # Train actor
        model.actor_optimizer.zero_grad()
        actor_loss.backward()
        model.actor_optimizer.step()

        if reward_list:
            # Train critic
            model.critic_optimizer.zero_grad()
            q1_loss.backward()
            model.critic_optimizer.step()

            model.critic2_optimizer.zero_grad()
            q2_loss.backward()
            model.critic2_optimizer.step()

        epoch_loss += actor_loss.data
        left, right = next_left, next_right

        if iter_ % 500 == 0:
            print('{} seconds to finished {}% cumulative loss is: {}'.format(time.time() - start, iter_ * 100.0 / max_iter, epoch_loss / iter_))     


def main():
    parser = argparse.ArgumentParser(description="Soft Actor Critic Network")
    parser.add_argument('-inputdim', type=int, dest='inputdim', help='input dimension')
    parser.add_argument('-hiddendim', type=int, dest='hiddendim', help='hidden dimension')
    parser.add_argument('-outputdim', type=int, dest='outputdim', help='output dimension')
    parser.add_argument('-bs', type=int, dest='bs', help='batch size')
    parser.add_argument('-lr', type=float, dest='lr', help='learning rate')
    #parser.add_argument('-decay', type=float, dest='decay', help='weight decay')
    parser.add_argument('-discount_rate', type=float, dest='discoutn_rate', help='discount_rate')
    parser.add_argument('-mod', type=str, dest='mod', help='mod') # ear crm

    A = parser.parse_args()
    print('Arguments loaded!')
    if A.mod == 'ear':
        inputdim = 89
    else:
        inputdim = 33
    PN = SAC-Net(input_dim=inputdim, dim1=A.hiddendim, output_dim=A.outputdim, alpha=A.lr, discout_rate=A.decay)

    cuda_(PN)
    print('Model on GPU')
    data_list = list()

    dir = '../../data/pretrain-numpy-data-{}'.format(A.mod)
    files = os.listdir(dir)
    file_paths = [dir + '/' + f for f in files]

    i = 0
    for fp in file_paths:
        with open(fp, 'rb') as f:
            try:
                data_list += pickle.load(f)
                i += 1
            except:
                pass
    print('total files: {}'.format(i))
    data_list = data_list[: int(len(data_list) / 1.5)]
    print('length of data list is: {}'.format(len(data_list)))

    random.shuffle(data_list)

    train_list = data_list[: int(len(data_list) * 0.7)]
    valid_list = data_list[int(len(data_list) * 0.7): int(len(data_list) * 0.9)]
    test_list = data_list[int(len(data_list) * 0.9):]
    print('train length: {}, valid length: {}, test length: {}'.format(len(train_list), len(valid_list), len(test_list)))
    sleep(1)  # let you see this

    if A.optim == 'Ada':
        optimizer = torch.optim.Adagrad(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(8):
        random.shuffle(train_list)
        model_name = '../../data/PN-model-{}/pretrain-sac-model.pt'.format(A.mod)
        train_sac(A.bs, train_list, valid_list, test_list, PN, epoch, model_path, reward_list = None)


if __name__ == '__main__':
    main()