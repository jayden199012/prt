import torchvision.models as models
import torch.nn as nn


def create_module(pretrained, num_layers, label_num, divide_by=1,
                  relu=0.1, drop_out=0.2):
    module_list = nn.ModuleList()
    # get all layers of the pretrained model
    back_bone = nn.Sequential(*list(pretrained.children())[:-1])
    in_features = list(back_bone[-1].modules())[-1].num_features
    if num_layers > 0:
        for i in range(num_layers):
            module = nn.Sequential()
            out_features = in_features/divide_by
            ln = nn.Linear(in_features, out_features, bias=False)
            bn = nn.batchnorm1d(out_features)
            module.add_module(f"fc_layer_{i}", ln)
            module.add_module(f"leaky_relu", nn.LeakyReLU(relu, inplace=True))
            module.add_module(f"batch_norm_(i):", bn)
            module.add_module(f"drop_out_{i}", nn.Dropout(p=drop_out))
            module_list.append(module)
        module = nn.Sequential()
        module.add_module(nn.Linear(out_features, label_num))
        module.add_module(nn.Sigmoid())
        module_list.append(module)
    return module_list

pretrained = models.densenet121(pretrained=False)

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
        self.modules_list = create_module(
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