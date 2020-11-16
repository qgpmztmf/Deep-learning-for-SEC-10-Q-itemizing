#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def load_new(self, path):
        state_dict = t.load(path)
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, t.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        
        self.load_state_dict(own_state)

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    reshape input tensor to the size of (batch_size,dim_length)
    """

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
