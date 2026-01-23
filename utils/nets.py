import torch as T
import torch
import torch.nn as nn

from utils.resnet_new import resnet14, resnet20, ResNet
# from utils.resnet_bn import resnet10 as resnet10_bn, ResNet as ResNetBN
from utils.resnet_bn_new import resnet14 as resnet14_bn, resnet20 as resnet20_bn, ResNet as ResNetBN

from utils.wrn import WideResNet, WideResNetBlock, WideResNetNoBN, initialize_wrn_fixup
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Union
from pathlib import Path
from contextlib import contextmanager


SUPPORTED_ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


def activation_factory(name: str) -> nn.Module:
    activation_key = name.lower()
    if activation_key not in SUPPORTED_ACTIVATIONS:
        supported = ', '.join(sorted(SUPPORTED_ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Supported activations: {supported}")
    return SUPPORTED_ACTIVATIONS[activation_key]()


def get_model_presets():
    model_presets = {
        'linear': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2
            }
        },
        'linear_s': {
            'type': 'linear',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'bias': True
            }
        },
        'linear_l': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'lin_tiny': {
            'type': 'linear',
            'params': {
                'hidden_dim': 2,
                'n_layers': 1
            }
        },
        'mlp': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'relu',
            }
        },
        'mlp2': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 2,
                'activation': 'relu',
            }
        },
        'mlp3': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 3,
                'activation': 'relu',
            }
        },
        'mlp_s': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'activation': 'relu',
            }
        },
        'mlp_l': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4,
                'activation': 'relu',
            }
        },

        'mlp_silu': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'silu',
            }
        },
        'mlp_tanh': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'tanh',
            }
        },


        'cnn': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
                'activation': 'relu',
            }
        },

        'cnn_silu': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
                'activation': 'silu',
            }
        },


        'resnet': {
            'type': 'resnet',
            'params': {},
        },
        'resnet_bn': {
            'type': 'resnet_bn',
            'params': {},
        },
        'wrn': {
            'type': 'wrn',
            'params': {
                'depth': 10,
                'width_factor': 2,
            },
        },
        'wrn_no_bn': {
            'type': 'wrn_no_bn',
            'params': {
                'depth': 10,
                'width_factor': 2,
            },
        },
    }
    return model_presets





class SquaredLoss(nn.modules.loss._Loss):
    '''
    Basically MSE, but doesn't average over the dimensions.
    With added support for sampling_vector (aka weighting of the samples, aka mask) the samples! 
    Used to do GD with noise to simulate SGD
    '''
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: T.Tensor, target: T.Tensor,
                sampling_vector: T.Tensor = None,
                reduction: str = None
                ) -> T.Tensor:
        '''
        THE MASK NEEDS TO BE "NORMALIZED" - i.e. with expected value of 1/n*unit_vector, 
        and NOT just unit_vector, without the normalization
        this is because without mask there is averaging happening!
        Inherently, this is to perform Jacobian-vector products
        '''

        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape for the loss to operate as expected.\nDid you forget to squeeze the output?")
        

        if sampling_vector is not None:
            total_L2 = F.mse_loss(input, target, reduction='none') 
            
            if len(target.shape) != 1:
                loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
            else:
                loss_per_sample = total_L2

            assert len(loss_per_sample.shape) == 1
            sampled_loss = T.dot(loss_per_sample, sampling_vector) # L \dot \omega - where \omega is the sampling vector
            return sampled_loss
        
        # if len(target.shape) != 1:
        #     # used to 
        #     multiplier = input.size(-1)
        # else:
        #     multiplier = 1.
        
        total_L2 =  F.mse_loss(input, target, reduction='none') # shape = (batch_size, num_classes)
        if len(target.shape) != 1:
            loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
        else:
            loss_per_sample = total_L2

        
        if not reduction is None:
            if reduction == 'none':
                return loss_per_sample

            raise ValueError(f"Are you sure you want to use reduction={reduction}? Double-check what you doing - maybe use self.reduction variable at __init__ instead?\n")
        
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        if self.reduction == 'sum':
            return loss_per_sample.sum()
        # we are not introducing this just as a safety
        # if self.reduction == 'none':
        #     return loss_per_sample
        
        raise ValueError("Unknown reduction type")
        



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, activation: str = 'relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.activation = activation_factory(self.activation_name)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def __repr__(self):
        return f"MLP({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim}, activation={self.activation_name})"


class CNN(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim, activation: str = 'relu'):
        super(CNN, self).__init__()
        self.fc_hidden_dim = fc_hidden_dim
        self.activation_name = activation.lower()

        def act():
            return activation_factory(self.activation_name)

        # self.conv1 = nn.Conv2d(3, 64, 3, 1)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.convs = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1), # 64*30*30
                act(),
                nn.Conv2d(64, 64, 3, 1), # 64*28*28
                act(),
                nn.MaxPool2d(2, 2), # 64, 14

                nn.Conv2d(64, 128, 3, 1), # 128, 12
                act(),
                nn.MaxPool2d(2, 2), # 128, 6
        )
        self.fcs = nn.Sequential(
                nn.Linear(128*6*6, fc_hidden_dim, bias=True),
                act(),
                nn.Linear(fc_hidden_dim, output_dim, bias=True)
        )
        # self.fc1 = nn.Linear(128*6*6, width, bias=False)
        # self.fc2 = nn.Linear(width, 1, bias=False)
        # self.apply(_weights_init)

    def forward(self, x):
        x = self.convs(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        x = self.fcs(x)
        return x


class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, bias=True):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Linear({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


def prepare_net(model_type: str,
                params: dict
                ):
    activation = params.get('activation', 'relu')
    if model_type == 'linear':
        net = Linear(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], params['bias'])
    
    if model_type == 'mlp':
        net = MLP(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], activation=activation)
    
    if model_type == 'cnn':
        net = CNN(params['hidden_dim'], params['output_dim'], activation=activation)
    
    if model_type == 'resnet':
        net = resnet20()
    
    if model_type == 'resnet_bn':
        net = resnet14_bn()

    if model_type == 'wrn':
        net = WideResNet(
            depth=params['depth'],
            widen_factor=params['width_factor'],
            num_classes=params['output_dim'],
        )
    
    if model_type == 'wrn_no_bn':
        net = WideResNetNoBN(
            depth=params['depth'],
            widen_factor=params['width_factor'],
            num_classes=params['output_dim'],
        )

    return net

def prepare_net_dataset_specific(model_name: str,
                                 dataset: str,
                                 ):
    '''
    Returns the model specific to the provided dataset
    '''
    from utils.data import get_dataset_presets

    model_presets = get_model_presets()
    params = model_presets[model_name]['params']
    model_type = model_presets[model_name]['type']

    dataset_presets = get_dataset_presets()
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']

    net = prepare_net(model_type, params)

    return net


def _init_linear_with_activation(module: nn.Linear, activation: str, scale: float):
    activation = activation.lower()
    if activation == 'tanh':
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    module.weight.data.mul_(scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _init_conv_with_activation(module: nn.Conv2d, activation: str, scale: float):
    activation = activation.lower()
    if activation == 'tanh':
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    module.weight.data.mul_(scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def initialize_mlp(net, scale=None):
    if scale is None:
        scale=1
    activation = getattr(net, 'activation_name', 'relu')
    for m in net.modules():
        if isinstance(m, nn.Linear):
            _init_linear_with_activation(m, activation, scale)


def initialize_cnn(net, scale=None):
    if scale is None:
        scale = 1.0
    activation = getattr(net, 'activation_name', 'relu')
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            _init_conv_with_activation(m, activation, scale)
        elif isinstance(m, nn.Linear):
            _init_linear_with_activation(m, activation, scale)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def initialize_resnet_old(net, scale=0.01):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data *= 0.1
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    net.fc.weight.data *= scale

def initialize_resnet(net, scale=None):
    if scale is None:
        scale = 1
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    # T.nn.init.normal_(net.linear.weight, std=scale)


def initialize_resnet_bn_old(net, scale=0.1):
    for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    # initialize fc layer
    nn.init.kaiming_normal_(net.fc.weight, mode='fan_out', nonlinearity='relu')
    # self.fc.weight.data.mul_(0.1)
    nn.init.constant_(net.fc.bias, 0)

    # # Zero-initialize the last BN in each residual branch,
    # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    # for m in net.modules():
    #     if isinstance(m, BasicBlock):
    #         pass
    #         nn.init.constant_(m.bn2.weight, 0)

    # custom scale
    net.fc.weight.data *= scale


def initialize_resnet_bn(net, scale=None, zero_init_residual=False):
    if scale is None:
        scale = 1

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    if not zero_init_residual:
        return

    for module in net.modules():
        # Prefer the deepest BN (bn3) when available, otherwise fall back to bn2.
        final_bn = None
        if hasattr(module, "bn3") and isinstance(module.bn3, nn.BatchNorm2d):
            final_bn = module.bn3
        elif hasattr(module, "bn2") and isinstance(module.bn2, nn.BatchNorm2d):
            final_bn = module.bn2

        if final_bn is not None:
            nn.init.zeros_(final_bn.weight)


def initialize_wrn(net: WideResNet, scale=None):
    if scale is None:
        scale = 1.0

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            ### NOTE
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    for block in net.modules():
        if isinstance(block, WideResNetBlock):
            nn.init.zeros_(block.bn2.weight)

    nn.init.kaiming_uniform_(net.fc.weight)
    #### NOTE THE SCALE!
    net.fc.weight.data.mul_(scale)
    if net.fc.bias is not None:
        nn.init.zeros_(net.fc.bias)


# def initialize_wrn_old(net, scale=None):
#     if scale is None:
#         scale = 1.0

#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             # m.weight.data.mul_(scale)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
#             # m.weight.data.mul_(scale)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
    

#     # custom scale
#     net.fc.weight.data *= scale



def initialize_linear(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


@contextmanager
def temp_seed(seed):
    '''
    Temporarily sets the seed for the random number generator
    This is a context
    '''
    if seed is None:
        yield
        return
        
    state = T.get_rng_state()
    T.manual_seed(seed)
    if T.cuda.is_available():
        cuda_state = T.cuda.get_rng_state()
        T.cuda.manual_seed(seed)
    
    try:
        yield
    finally:
        T.set_rng_state(state)
        if T.cuda.is_available():
            T.cuda.set_rng_state(cuda_state)


def initialize_net(net, scale=None, seed=None):

    with temp_seed(seed):
        if isinstance(net, Linear):
            initialize_linear(net, scale=scale)
        elif isinstance(net, MLP):
            initialize_mlp(net, scale=scale)    
        elif isinstance(net, ResNet):
            initialize_resnet(net, scale=scale)
        elif isinstance(net, ResNetBN):
            initialize_resnet_bn(net, scale=scale, zero_init_residual=False)
        elif isinstance(net, CNN):
            initialize_cnn(net, scale=scale)
        elif isinstance(net, WideResNet):
            initialize_wrn(net, scale=scale)
        elif isinstance(net, WideResNetNoBN):
            initialize_wrn_fixup(net)
        else:
            raise ValueError("Unknown net type")


def prepare_optimizer(net, lr, momentum, adam):
    if adam:
        if momentum is not None:
            raise ValueError("Momentum is not supported for Adam, just because. Change the code if you need to change the params in Adam")
        return T.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    if momentum is not None:
        return T.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    return T.optim.SGD(net.parameters(), lr=lr, momentum=0)


def get_path_of_last_net(path: Union[str, Path], not_final=False):
    path = Path(path)
    if path.is_dir():
        files = list(path.glob('*.pt'))
        if 'net_final.pt' in [file.name for file in files]:
            return path / 'net_final.pt'
        if len(files) == 0:
            return None
        files.sort(key=lambda x: x.stat().st_mtime)

        return files[-1]
    else:
        return path
