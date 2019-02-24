import torch
import torch.nn as nn
import numpy as np


def weights_init(m, stdv=0.05):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-stdv, stdv)
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.zero_()


class empty_layer(nn.Module):
    def __init__(self):
        super().__init__()


def special_layer(module, layer_index, layer_name):
    scl = empty_layer()
    module.add_module(f"{layer_name}_{layer_index}", scl)


def create_module(input_dim, output_dim, label_num, num_of_layers,
                  start, step):
    module_list = nn.ModuleList()
    short_cut_layers = list(np.arange(start=start,
                                      stop=num_of_layers,
                                      step=step))
    for i in range(num_of_layers):
        module = nn.Sequential()
        if i not in short_cut_layers:
            ln = nn.Linear(input_dim, output_dim, bias=False)
            bn = nn.BatchNorm1d(output_dim)
            input_dim = output_dim
            module.add_module(f"hidden_layer_{i}", ln)
            module.add_module(f"LeakyReLU_{i}", nn.LeakyReLU())
            module.add_module(f"bach_norm_{i}", bn)
            module.add_module(f"dropout_{i}", nn.Dropout(p=0.2))
        else:
            special_layer(module, i, 'scl')
        module_list.append(module)
    module_list.append(nn.Linear(output_dim, label_num))
    module_list.append(nn.Sigmoid())
    return module_list, short_cut_layers


class Ordinal_regression(nn.Module):
    def __init__(self, create_module, config, input_dim):
        super().__init__()
        self.config = config.copy()
        self.input_dim = input_dim
        self.output_dim = self.config['output_dim']
        self.label_num = self.config['label_num']
        self.num_of_layers = self.config['num_of_layers']
        self.start = self.config['start']
        self.step = self.config['step']
        self.modules_list, self.short_cut_layers = create_module(
                                                            self.input_dim,
                                                            self.output_dim,
                                                            self.label_num,
                                                            self.num_of_layers,
                                                            self.start,
                                                            self.step)
        self.referred_layers = [x - self.step for x in self.short_cut_layers]
        self.bce_loss = nn.BCELoss()
        if self.config['pretrain_snapshot']:
            state_dic = torch.load(self.config['pretrain_snapshot'])
            self.load_state_dict(state_dic)
        else:
            self.apply(weights_init)

    def forward(self, x, cuda, is_training=False, labels=None):
        if cuda:
            self.bce_loss = self.bce_loss.cuda()
        for index, layer in enumerate(self.modules_list):
            if index not in self.short_cut_layers:
                x = self.modules_list[index](x)
                if index in self.referred_layers:
                    cache = x
            else:
                x += cache
        if is_training:
            loss = self.bce_loss(torch.sqrt(x+1e-16), torch.sqrt(labels))
            return loss
        else:
            prediction = torch.sum(x.round(), dim=1) + 1
            return prediction
