from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torchviz import make_dot


# class DuelingNetwork(nn.Module):
#
#     def __init__(self, obs, ac):
#         super().__init__()
#
#         self.model = nn.Sequential(nn.Linear(obs, 128),
#                                    nn.ReLU(),
#                                    nn.Linear(128, 128),
#                                    nn.ReLU())
#
#         self.value_head = nn.Linear(128, 1)
#         self.adv_head = nn.Linear(128, ac)
#
#     def forward(self, x):
#         out = self.model(x)
#
#         value = self.value_head(out)
#         adv = self.adv_head(out)
#
#         q_val = value + adv - adv.mean(1).reshape(-1, 1)
#         return q_val


class BranchingQNetwork(nn.Module):

    def __init__(self, observation_size, action_dim, n):
        super().__init__()

        self.ac_dim = action_dim
        self.n = n                                            # number of bins for discretised action (same in each dim/branch)

        self.model = nn.Sequential(nn.Linear(observation_size, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(action_dim)])

    def forward(self, x):
        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        # shape of [l(out) for l in self.adv_heads] is action_dim * torch.Size([observation_size, n])
        # print(advs.shape) # torch.Size([observation_size, action_dim, n])
        # print(advs.mean(2).shape) # torch.Size([observation_size, action_dim])
        # print(advs.mean(2, keepdim=True).shape) # torch.Size([observation_size, action_dim, 1])
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)
        # print(q_val.shape) # torch.Size([observation_size, action_dim]) # torch.Size([observation_size, action_dim,n])

        return q_val

    @staticmethod
    def state_dict_from_weights(model: nn.Module, weights):
        """ Helper method to transform model parameters into a loadable state dict"""
        return OrderedDict([(k, v) for k, v in zip(model.state_dict().keys(), weights)])

    def soft_update(self, other_model: "BranchingQNetwork", tau: float = None):
        """ Updates model parameters using a weighted sum with parameters of other_model controlled by tau.
            Expected usage: should be called on target model, other_model being online model.
            θ_target = tau * θ_online + (1 - tau) * θ_target """
        if tau is None:
            self.model.load_state_dict(other_model.state_dict())
        else:
            new_target_weights = []
            for target, online in zip(self.model.state_dict().values(), other_model.model.state_dict().values()):
                target_ratio = (1.0 - tau) * target
                online_ratio = tau * online
                mixed_weights = target_ratio + online_ratio
                new_target_weights.append(mixed_weights)
            new_state_dict = BranchingQNetwork.state_dict_from_weights(self.model, new_target_weights)
            self.model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    b = BranchingQNetwork(
        observation_size=5,
        action_dim=4,
        n=6
    )
    print(b)  # funktioniert mit torchviz

    x = torch.rand(10, 5)
    print(x.shape)  # torch.Size([10, 5])
    yhat = b(x)
    print(yhat.shape)  # torch.Size([10, 4, 6])

    # from torchsummary import summary
    # summary(b, (5, 128)) # (to be fixed)


