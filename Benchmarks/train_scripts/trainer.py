import torch
import torch.nn as nn
from torch.optim import Adam
import time


class Trainer:
    def __init__(self,
                 net: nn.Module,
                 lr:float,
                 n_epochs: int,
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.net = net
        self.net = self.net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.n_epochs = n_epochs



    def compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        return
    def run(self):
        return
