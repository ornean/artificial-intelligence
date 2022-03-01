from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torchviz import make_dot


class DuelingNetwork(nn.Module):

    def __init__(self, obs, ac):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x):
        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1, 1)
        return q_val

    @staticmethod
    def state_dict_from_weights(model: nn.Module, weights):
        """ Helper method to transform model weight parameters into a loadable state dict"""
        try:
            return OrderedDict([(k, v) for k, v in zip(model.state_dict().keys(), weights)])
        except ValueError:
            print("Error while transferring weights")

    def soft_update(self, other_model: "DuelingNetwork", tau=None):
        """ Updates model weights from weights of another model using a weighted sum controlled by tau.
            Expected usage: model is target model and other_model online model.
            new model weights = (1 - tau) * model weights + tau * other model weights"""
        if tau is None:
            self.model.load_state_dict(other_model.state_dict())
        else:
            new_target_weights = []
            for target, online in zip(self.model.state_dict().values(), other_model.model.state_dict().values()):
                target_ratio = (1.0 - tau) * target
                online_ratio = tau * online
                mixed_weights = target_ratio + online_ratio
                new_target_weights.append(mixed_weights)
            new_state_dict = DuelingNetwork.state_dict_from_weights(self.model, new_target_weights)
            self.model.load_state_dict(new_state_dict)


# class BranchingQNetwork(nn.Module):
#
#     def __init__(self, obs, ac_dim):
#         super().__init__()
#
#         self.ac_dim = ac_dim
#
#         self.model = nn.Sequential(nn.Linear(obs, 128),
#                                    nn.ReLU(),
#                                    nn.Linear(128, 128),
#                                    nn.ReLU())
#
#         self.value_head = nn.Linear(128, 1)
#         self.adv_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(ac_dim)])
#
#     def forward(self, x):
#         out = self.model(x)
#         value = self.value_head(out)
#         advs = torch.cat([l(out) for l in self.adv_heads], dim=0)
#
#         # print(advs.mean(2).shape)
#         # test = advs.mean(2, keepdim=True)
#         # input(test.shape)
#         q_val = (value + advs.unsqueeze(2) - advs.mean(0, keepdim=True)).squeeze()
#         # input(q_val.shape)
#
#         return q_val


if __name__ == "__main__":
    b = DuelingNetwork(5, 4)

    # print(b)  # funktioniert mit torchviz
    yhat = b(torch.rand(128, 5))

    print(yhat.shape)

    # from torchsummary import summary
    # summary(b, (5, 128))
