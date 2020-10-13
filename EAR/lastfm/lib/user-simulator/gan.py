import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super(Generator, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            nn.ReLU()
            nn.Linear(hidden_dim, output_dim)
            nn.Tanh()
        )
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

    def forward(self, input):
        assert input.shape[-1] == self.input_dim, "Input dimension doesn't match!"
        out = self.layers(input)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super(Discriminator, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            nn.ReLU()
            nn.Linear(hidden_dim, output_dim)
            nn.Sigmoid()
        )
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')

    def forward(self, input):
        assert input.shape[-1] == self.input_dim, "Input dimension doesn't match!"
        out = self.layers(input)
        return out


def train_GAN(G, D, train_list, disc_fake_data, gen_fake_data,
              G_optimizer, D_optimizer, num_epochs, 
              g_steps, d_steps, bs, model_name):
    '''
    params: 
    '''
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        left, right = epoch * bs, min(len(train_list), (epoch + 1) * bs)
        data_batch = train_list[left: right]

        a = [item[1] for item in data_batch]
        s = a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()

        temp_in = cuda_(temp_in)

        for i in range(d_steps):
            # Train Discriminator on real state data
            d_real_decision = D(train_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1])))
            d_real_error.backward()

            # Train Discriminator on fake state data
            d_fake_data = G(fake_data).detach() 
            d_fake_decision = D(d_fake_data.t())
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))
            d_fake_error.backward()
            D_optimizer.step()     
            D_optimizer.zero_grad()

        for j in range(g_steps):
            g_fake_data = G(gen_fake_data)
            dg_fake_decision = D(g_fake_data.t())
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1])))  

            g_error.backward()
            G_optimizer.step()
            G_optimizer.zero_grad()
        # TODO: add print statements

        if epoch % 7 == 0:
            PATH = model_path
            torch.save(G.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

def main():
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument('-inputdim', type=int, dest='inputdim', help='input dimension')
    parser.add_argument('-hiddendim', type=int, dest='hiddendim', help='hidden dimension')
    parser.add_argument('-outputdim', type=int, dest='outputdim', help='output dimension')
    parser.add_argument('-bs', type=int, dest='bs', help='batch size')
    parser.add_argument('-g_lr', type=float, dest='g_lr', help='generator learning rate')
    parser.add_argument('-d_lr', type=float, dest='d_lr', help='discriminator learning rate')
    #parser.add_argument('-g_decay', type=float, dest='g_decay', help='weight decay for generator')
    #parser.add_argument('-d_decay', type=float, dest='d_decay', help='weight decay for discriminator')
    parser.add_argument('-mod', type=str, dest='mod', help='mod') # ear crm
    parser.add_argument('-num_epochs', type=int, dest='num_epochs', help='number of training epochs')
    parser.add_argument('-d_steps', type=int, dest='d_steps', help='number of training steps in each epoch of generator')
    parser.add_argument('-g_steps', type=int, dest='g_steps', help='number of training steps in each epoch of discriminator')

    A = parser.parse_args()
    print('Arguments loaded!')
    if A.mod == 'ear':
        inputdim = 89
    else:
        inputdim = 33
    G = Generator(inputdim, A.hiddendim, A.outputdim, A.g_lr)
    D = Discriminator(inputdim, A.hiddendim, A.outputdim, A.d_lr)

    cuda_(G)
    cuda_(D)
    print('Model on GPU')

    data_list = list()

    np_dir = '../../data/pretrain-sac-numpy-data-{}'.format(A.mod)
    files = os.listdir(np_dir)
    file_paths = [np_dir + '/' + f for f in files]     
    
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

    data_list = data_list[: int(len(data_list) / 1.5)]
    train_data = data_list[: int(len(data_list) * 0.7)]
    g_fake_data = cuda_(torch.rand(A.bs, inputdim))
    d_fake_data = cuda_(torch.normal(0, 1, size = (bs, inputdim)))
    d_optimizer = optim.SGD(D.parameters(), lr=D.lr)
    g_optimizer = optim.SGD(G.parameters(), lr=G.lr)
    model_name = '../../data/gan-model-{}/pretrain-gan-model.pt'.format(A.mod)

    train_GAN(G, D, train_data, d_fake_data, g_fake_data,
              g_optimizer, d_optimizer, A.num_epochs, 
              A.g_steps, A.d_steps, model_name)
