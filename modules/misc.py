import torch.nn as nn
from inspect import isfunction


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):
    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.cuda(non_blocking=True))


def get_activation_layer(activation):
    
    """
    Create activation layer from string/function.
    ----------
    activation : function, or str, or nn.Module | Activation function or name of activation function
    Returns    : nn.Module                      | Activation layer
    """
    
    assert (activation is not None)
    
    if isfunction(activation):
        return activation()
    
    elif isinstance(activation, str):
        
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return HSigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError()
            
    else:
        assert (isinstance(activation, nn.Module))
        return activation