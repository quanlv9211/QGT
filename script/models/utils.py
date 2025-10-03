import math
import torch.nn as nn

def get_activation(name):
    """
    Returns a PyTorch activation function based on the provided name.
    
    Args:
        name (str): Name of the activation function. Options:
                    'relu', 'sigmoid', 'tanh', 'leakyrelu', 'gelu', 'elu'
    
    Returns:
        nn.Module: The corresponding PyTorch activation module
    
    Raises:
        ValueError: If an unsupported activation name is provided.
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
